"""Generate every paper table, figure source, and paired test from saved results."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from .metrics import load_rows, source_metrics
from .protocol import (CONDITIONS, PARENT_COUNTS, correctness, file_sha256,
                       historical_tree_identity, validate_rows, write_json, object_sha256)
from .suite import load_experiments, validate_run_spec

from .artifacts import MAIN
ABLATION = ["public_child", "repro_child", "child_child", "public_original", "repro_original", "child_original"]


def bundled_rows(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = json.loads((root / "evaluation_sources.json").read_text())
    rows, missing = {}, {}
    for name in CONDITIONS:
        if name not in manifest:
            missing[name] = "No identified complete historical output; manuscript_only values are not used"
            continue
        path = root / "outcomes" / (name + ".jsonl")
        if file_sha256(path) != manifest[name]["outcomes_sha256"]:
            raise ValueError("Frozen historical outcome hash mismatch: " + name)
        values = load_rows(path)
        validate_rows(values, expected)
        rows[name] = values
    numeric = root / "decisions/public_child.jsonl"
    audit = json.loads((root / "public_rank_reaggregation.json").read_text())
    fingerprints = json.loads((root / "artifact_fingerprints.json").read_text())
    if audit["source_test_sha256"] != fingerprints["source_test"]["sha256"]:
        raise ValueError("Expected population was not exported from the historical test input")
    if file_sha256(numeric) != audit["numeric_decisions_sha256"]:
        raise ValueError("Public numeric decision fingerprint mismatch")
    if audit["raw_results_sha256"] != manifest["public_child"]["raw_results_sha256"]:
        raise ValueError("Public decision/raw provenance mismatch")
    public = load_rows(numeric)
    validate_rows(public, expected, parent_counts=PARENT_COUNTS)
    if {r["source_id"]: correctness(r) for r in public} != {r["source_id"]: correctness(r) for r in rows["public_child"]}:
        raise ValueError("Public numeric decision correctness differs from frozen outcomes")
    # Verify the legacy ranks too, before using numeric evidence for corrected ranks.
    for depth, metric in source_metrics(public)["sample_level_metrics"].items():
        compact_metric = source_metrics(rows["public_child"])["sample_level_metrics"][depth]
        for field in ("accuracy_any", "topk_any_rate", "mean_valid_rank", "median_valid_rank"):
            if metric[field] != compact_metric[field]:
                raise ValueError("Public legacy rank parity mismatch: " + field)
    rows["public_child"] = public
    return rows, expected, missing, {"kind": "frozen historical outcomes", "weight_identity_verified": False}


def suite_rows(directory, experiments):
    directory = Path(directory)
    if not (directory / 'suite.json').is_file():
        raise ValueError('Supply a completed --suite-dir containing suite.json and condition outputs')
    plan = json.loads((directory / "suite.json").read_text())
    if plan["profile"] not in {"historical", "corrected-ranks"}:
        raise ValueError("Unknown suite profile")
    expected = json.loads((directory / "inputs.json").read_text())
    if expected != plan["inputs"]:
        raise ValueError("Suite/source manifest mismatch")
    rows, missing, code_versions = {}, {}, set()
    for name in CONDITIONS:
        if name not in plan["conditions"] or not (directory / name / "results.jsonl").exists():
            missing[name] = "Condition output is missing"
            continue
        spec = plan["conditions"][name]
        if spec["condition_id"] != name:
            raise ValueError("Condition ID mismatch")
        protocol = json.loads((directory / name / "protocol.json").read_text())
        invocation = json.loads((directory / name / "invocation.json").read_text())
        if protocol["run_spec"] != spec:
            raise ValueError("Condition protocol differs from suite manifest")
        for key in ("model", "prompt", "max_tokens", "context_length", "historical_context_length"):
            if spec[key] != experiments["conditions"][name][key]:
                raise ValueError("Condition settings differ from experiment manifest: " + key)
        for key in ("top_logprobs", "api_max_retries", "format_max_retries", "extractor_version"):
            if spec[key] != experiments["evaluation"][key]:
                raise ValueError("Evaluation settings differ from experiment manifest: " + key)
        policy = "legacy" if plan["profile"] == "historical" else "none"
        if spec["format_error_ranks"] != policy or protocol["format_error_ranks"] != policy:
            raise ValueError("Mixed rank policies in suite")
        validate_run_spec(spec, SimpleNamespace(**invocation), protocol["tree_identity"], expected)
        if protocol["tree_identity"] != plan["tree_identity"]:
            raise ValueError("Suite tree identity mismatch")
        values = load_rows(directory / name / "results.jsonl")
        validate_rows(values, expected, expected_tree=plan["tree_identity"], actual_tree=protocol["tree_identity"],
                      parent_counts=PARENT_COUNTS if plan["tree_profile"] == "historical" else None)
        rows[name] = values
        code_versions.add((protocol["prompt_store_sha256"], protocol["extractor_source_sha256"]))
    if len(code_versions) > 1:
        raise ValueError("Mixed prompt/extractor source versions")
    declarations = {}
    for name in rows:
        spec = plan["conditions"][name]
        declaration = (spec["checkpoint"], spec["model_name"])
        model = spec["model"]
        if model in declarations and declarations[model] != declaration:
            raise ValueError("One model has inconsistent checkpoint declarations")
        declarations[model] = declaration
    if len({d[0] for d in declarations.values()}) != len(declarations) or len({d[1] for d in declarations.values()}) != len(declarations):
        raise ValueError("Different models share a checkpoint or served name")
    # Paired formats must use the same declared checkpoint and served model identity.
    for first, second in [("public_child", "public_original"), ("repro_child", "repro_original"), ("child_child", "child_original")]:
        if first in rows and second in rows:
            a, b = plan["conditions"][first], plan["conditions"][second]
            if (a["checkpoint"], a["model_name"]) != (b["checkpoint"], b["model_name"]):
                raise ValueError("Paired formats used different model declarations")
    return rows, expected, missing, {"kind": "suite results", "plan": plan,
                                   "weight_identity_verified": False}


def emit_table(directory, name, rows):
    if not rows:
        return
    fields = list(rows[0])
    with (directory / (name + ".csv")).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: "NA" if value is None else value for key, value in row.items()} for row in rows)

    def cell(value):
        if value is None:
            return "NA"
        return str(value).replace("_", r"\_").replace("%", r"\%")

    lines = [r"\begin{tabular}{" + "l" * len(fields) + "}", " & ".join(map(cell, fields)) + r" \\", r"\hline"]
    lines += [" & ".join(cell(row[k]) for k in fields) + r" \\" for row in rows]
    lines.append(r"\end{tabular}")
    (directory / (name + ".tex")).write_text("\n".join(lines) + "\n")


def accuracy_row(name, values):
    row = {"condition": name}
    accuracies = [100 * sum(correctness(r)[d] for r in values) / len(values) for d in range(5)] if values else [None]*5
    row.update({f"D{i}": round(v, 2) if v is not None else None for i, v in enumerate(accuracies)})
    row["Overall"] = round(sum(accuracies)/5, 2) if values else None
    row["Macro_D0_D3"] = round(sum(accuracies[:4])/4, 2) if values else None
    row["n"] = len(values) if values else None
    row["evidence"] = "saved_outcomes" if values else "missing; no numeric substitution"
    return row


def topk_rows(metrics):
    result = []
    for depth, m in metrics["sample_level_metrics"].items():
        row = {"depth": "D" + depth, "Top1": round(100*m["accuracy_any"], 2)}
        row.update({f"Top{k}": round(100*m["topk_any_rate"][f"top_{k}"], 2) for k in [2, 3, 5]})
        row.update(Mean=round(m["mean_valid_rank"], 2) if m["mean_valid_rank"] is not None else None,
                   Median=m["median_valid_rank"], valid_rank_n=m["valid_rank_sources"], n=m["n"],
                   rank_policy=metrics["rank_policy"])
        result.append(row)
    return result


def training_settings(config, project_root):
    """Read the scalar optimization values from the actual two portable configs."""
    def read(path):
        result = {}
        for line in path.read_text().splitlines():
            if ':' not in line or line.lstrip().startswith('#'):
                continue
            key, value = line.split(':', 1)
            value = value.strip().strip('"')
            try:
                result[key] = json.loads(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
        return result
    a = read(Path(project_root) / config['training']['paper_selection'])
    b = read(Path(project_root) / config['training']['child_subtree'])
    keys = ['seed', 'num_train_epochs', 'learning_rate', 'warmup_ratio', 'weight_decay',
            'per_device_train_batch_size', 'gradient_accumulation_steps', 'bf16', 'finetuning_type',
            'lr_scheduler_type', 'cutoff_len', 'max_grad_norm', 'flash_attn', 'deepspeed']
    if any(a[k] != b[k] for k in keys):
        raise ValueError('Paper/Child optimization configs differ')
    values = {k: a[k] for k in keys}
    if a['model_name_or_path'] != b['model_name_or_path'] or a['model_name_or_path'] != config['training']['backbone']:
        raise ValueError('Training backbone differs from experiment manifest')
    zero = json.loads((Path(project_root) / a['deepspeed']).read_text())['zero_optimization']
    if zero['stage'] != 3 or zero['offload_optimizer']['device'] != 'cpu' or zero['offload_param']['device'] != 'cpu':
        raise ValueError('Expected ZeRO-3 CPU offload configuration')
    values.update(backbone=a['model_name_or_path'], deepspeed='ZeRO-3 CPU offload')
    return values


def validate_artifact_files(directory, planned):
    """Only existing, nonempty generated files may be described as generated."""
    root = Path(directory)
    generated, missing = {}, {}
    for label, files in planned.items():
        present = [f for f in files if (root / f).is_file() and (root / f).stat().st_size]
        generated[label] = present
        absent = [f for f in files if f not in present]
        if absent:
            missing[label] = {'status': 'missing_artifact', 'files': absent, 'reason': 'Required artifact was not generated (missing input or explicitly skipped computation).'}
    return generated, missing


def generate_report(output, *, results_root=None, suite_dir=None, experiments='configs/experiments.json',
                    tree_summary=None, training_audit=None, corpus_audit=None, project_root=None,
                    claim_manifest=None, allow_partial=False, iterations=5000, seed=42, skip_statistics=False):
    from .artifacts import validate_inputs, population_gaps, validate_figure_assets, make_figures
    from .claims import INVENTORY_SHA256, validate_manifest, evaluate_claims, save_claims
    output, root = Path(output), Path(results_root) if results_root else Path('results')
    if not suite_dir and results_root is None:
        raise ValueError('Supply --suite-dir with --tree-summary, --training-audit and --corpus-audit; historical results are not bundled')
    project = Path(project_root) if project_root else Path(experiments).resolve().parent.parent
    manifest_path = Path(claim_manifest) if claim_manifest else project / 'configs/output_schema.json'
    manuscript = json.loads(manifest_path.read_text())
    validate_manifest(manuscript)
    if output.exists():
        raise FileExistsError(output)
    config = load_experiments(experiments)
    if suite_dir:
        if not tree_summary or not training_audit or not corpus_audit:
            raise ValueError('Suite reports require explicit --tree-summary, --training-audit and --corpus-audit')
        rows, expected, missing, provenance = suite_rows(suite_dir, config)
        identity = provenance['plan']['tree_identity']
    else:
        rows, expected, missing, provenance = bundled_rows(root)
        fp = json.loads((root / 'artifact_fingerprints.json').read_text())
        identity = historical_tree_identity(fp)
    tree_path = Path(tree_summary) if tree_summary else root / 'tree_granularity.json'
    audit_path = Path(training_audit) if training_audit else root / 'training_source_audit.json'
    corpus_path = Path(corpus_audit) if corpus_audit else root / 'corpus_audit.json'
    tree = json.loads(tree_path.read_text())
    training = json.loads(audit_path.read_text())
    data = json.loads(corpus_path.read_text()) if corpus_path.exists() else None
    validate_inputs(tree, training, data, identity, expected['count'],
                    historical=not suite_dir or provenance['plan']['tree_profile'] == 'historical',
                    required_training_examples=config['training']['examples_per_model'])
    if not suite_dir:
        if data and (data['inputs']['source_test_sha256'] != fp['source_test']['sha256'] or
                     data['inputs']['source_train_sha256'] != fp['source_train']['sha256']):
            raise ValueError('Historical corpus input fingerprint mismatch')
        for key in ('source_train', 'paper_training_prefix', 'child_training_prefix'):
            if training['provenance']['inputs'][key] != fp[key]['sha256']:
                raise ValueError('Historical training audit fingerprint mismatch')
    elif expected['mode'] != provenance['plan']['mode']:
        raise ValueError('Suite population mode mismatch')
    if data is None:
        missing['corpus_audit'] = 'No validated raw-corpus and leaf-path audit supplied'
    elif data.get('source_sets', {}).get('original_ids_sha256' if suite_dir else 'pseudonymous_ids_sha256') != expected['source_ids_sha256']:
        raise ValueError('Corpus audit source set differs from evaluation population')
    if data is not None:
        missing.update(population_gaps(data, expected['count']))
    if expected['mode'] != 'paper':
        missing['paper_population'] = 'Toy evaluation cannot claim a complete paper experiment'
    if skip_statistics:
        missing['paired_statistics'] = 'Explicitly skipped; report is partial'
    assets = validate_figure_assets(project)
    # Reject known structural incompleteness before importing optional analysis
    # dependencies. This also works in a standard-library-only installation.
    if missing and not allow_partial:
        raise ValueError('Incomplete paper report: ' + '; '.join(f'{k}: {v}' for k,v in missing.items()))
    stats = None
    if not skip_statistics and {'child_child', 'repro_original'}.issubset(rows):
        from .analysis import compare
        stats = compare({r['source_id']: correctness(r) for r in rows['child_child']},
                        {r['source_id']: correctness(r) for r in rows['repro_original']}, iterations, seed)
    elif not skip_statistics:
        missing['paired_statistics'] = 'Both main comparison outcomes are required'
    accuracy = {n: accuracy_row(n, rows.get(n)) for n in CONDITIONS}
    training = dict(training, shared_percent=100*training['shared_sources']/training['child_subtree_unique_sources'])
    settings = training_settings(config, project)
    settings.update(pub_training_records=data['counts']['train_rows'] if data else None,
        embedding_dim=data['tree_config'].get('embedding_dim') if data else None,
        branching_factor=data['tree_config'].get('branching_factor') if data else None,
        max_depth=data['tree_config'].get('max_depth') if data else None,
        representative='medoid' if data and data['tree_config'].get('use_medoid') == 1 else None,
        top_logprobs=config['evaluation']['top_logprobs'],
        base_max_tokens=config['conditions']['base_child']['max_tokens'],
        fine_tuned_max_tokens=config['conditions']['child_child']['max_tokens'])
    topk_legacy, topk_corrected = [], []
    if 'public_child' in rows:
        public = rows['public_child']
        if not suite_dir or provenance['plan']['profile'] == 'historical':
            topk_legacy = topk_rows(source_metrics(public, 'legacy'))
        topk_corrected = topk_rows(source_metrics(public, 'none'))
    context = {'accuracy': accuracy, 'tree': tree, 'training': training, 'corpus': data or {},
        'settings': settings, 'statistics': stats['statistics'] if stats else {},
        'topk_corrected': topk_corrected, 'topk_legacy': topk_legacy}
    claims = evaluate_claims(manuscript, context, historical=not suite_dir)
    for c in claims:
        if c['status'] == 'unverified':
            missing['claim:' + c['id']] = c['reason']
    if missing and not allow_partial:
        raise ValueError('Incomplete paper report: ' + '; '.join(f'{k}: {v}' for k,v in missing.items()))
    output.mkdir(parents=True)
    emit_table(output, 'all_conditions', list(accuracy.values()))
    emit_table(output, 'ablation', [accuracy[n] for n in ABLATION])
    emit_table(output, 'main_comparison', [accuracy[n] for n in MAIN])
    figure = make_figures(output, accuracy, assets, project)
    emit_table(output, 'depthwise', figure)
    if topk_legacy:
        emit_table(output, 'public_topk_legacy', topk_legacy)
    if topk_corrected:
        emit_table(output, 'public_topk_corrected', topk_corrected)
        write_json(output / 'public_topk_corrected.json', source_metrics(rows['public_child'], 'none'))
    emit_table(output, 'tree_granularity', [{'depth': 'D'+d, **v} for d,v in tree['depths'].items()])
    training_fields = ('paper_selection_examples', 'child_subtree_examples', 'paper_selection_unique_sources',
                       'child_subtree_unique_sources', 'shared_sources', 'shared_percent')
    emit_table(output, 'training_coverage', [{k: training[k] for k in training_fields}])
    emit_table(output, 'training_depth_distribution', [{'depth': 'D'+d, 'examples': n}
               for d,n in training['child_depth_distribution'].items()])
    def setting_evidence(key):
        if key in {'pub_training_records', 'embedding_dim', 'branching_factor', 'max_depth', 'representative'}:
            return 'raw corpus / saved evaluation tree audit'
        if key in {'top_logprobs', 'base_max_tokens', 'fine_tuned_max_tokens'}:
            return 'experiment manifest; historical settings qualifications below'
        return 'portable SFT configuration; GPU execution unverified'
    implementation = [{'item': k, 'setting': v, 'evidence': setting_evidence(k)} for k,v in settings.items()]
    implementation += [
        {'item': 'Training coverage', 'setting': '; '.join(f'{k}={training[k]}' for k in training_fields), 'evidence': training['provenance']['kind'] + '; weight identity unverified'},
        {'item': 'Training examples', 'setting': 'First ordered paper-selection prefix; ground-truth-path child decisions; final epoch checkpoint-3334', 'evidence': 'Two independent 10000-example SFT recipes; no held-out checkpoint selection'},
        {'item': 'Evaluation population', 'setting': str(expected['count']), 'evidence': expected['mode']},
        {'item': 'Evaluation tree papers', 'setting': tree['num_papers'], 'evidence': 'supplied tree identity; shared across conditions'},
        {'item': 'Training tree papers / root children', 'setting': 'See the supplied training artifacts; historical tree is not assumed', 'evidence': 'historical reference' if not suite_dir else 'operator-supplied training audit'},
        {'item': 'Training budget / devices', 'setting': f"{config['training']['examples_per_model']} examples/model; {config['training']['gpus']} GPUs", 'evidence': 'declared portable recipe; GPU execution unverified'},
        {'item': 'Generation by condition', 'setting': '; '.join(n+'='+str(s['max_tokens']) for n,s in config['conditions'].items()), 'evidence': 'historical Base 512; fine-tuned recipes 2048; repro_child historical unverified'},
        {'item': 'Contexts by condition', 'setting': '; '.join(n+'='+str(s['context_length']) for n,s in config['conditions'].items()), 'evidence': 'historical Base context unknown; repro_child context is recipe only'},
        {'item': 'Correctness', 'setting': 'source-record OR over positive parent decisions; independent local depths', 'evidence': 'protocol; no end-to-end traversal'}]
    emit_table(output, 'implementation', implementation)
    if data:
        emit_table(output, 'corpus_counts', [{'field': k, 'value': v} for k,v in data['counts'].items()])
        emit_table(output, 'parent_counts', [{'depth': 'D'+d, 'parent_decisions': n} for d,n in data['parent_counts'].items()])
    if stats is not None:
        write_json(output / 'main_paired_statistics.json', stats)
        emit_table(output, 'paired_tests', [{'depth': d, **{k:v for k,v in m.items() if k != 'ci95_pp'},
                   'ci95_low_pp': m['ci95_pp'][0], 'ci95_high_pp': m['ci95_pp'][1]}
                   for d,m in stats['statistics'].items() if d.startswith('D')])
        differences = [{'comparison': d, 'a_percent': 100*m['a_accuracy'], 'b_percent': 100*m['b_accuracy'],
                        'difference_pp': m['difference_pp'], 'exact_mcnemar_p': m['exact_mcnemar_p'], 'p_lt_001': m['exact_mcnemar_p'] < .001}
                       for d,m in stats['statistics'].items() if d.startswith('D')]
        differences += [{'comparison': d, 'a_percent': 100*m['a'], 'b_percent': 100*m['b'],
                         'difference_pp': m['difference_pp'], 'exact_mcnemar_p': None, 'p_lt_001': None}
                        for d,m in stats['statistics'].items() if d.startswith('macro')]
        emit_table(output, 'text_comparisons', differences)
    ci_claims = [c for c in claims if c['id'].startswith('ci.')]
    emit_table(output, 'ci_comparison', [{'claim': c['id'], 'manuscript': c.get('manuscript_value', 'not provided'),
        'recomputed': c['recomputed_value'], 'status': c['status']} for c in ci_claims])
    write_json(output / 'computed_values.json', context)
    claim_counts = save_claims(output, manuscript, claims)
    planned = {
        'tab:implementation': ['implementation.csv','implementation.tex','training_coverage.csv','training_depth_distribution.csv'],
        'tab:tree_granularity': ['tree_granularity.csv','tree_granularity.tex'],
        'tab:sample_level': ['public_topk_corrected.csv','public_topk_corrected.tex','public_topk_corrected.json'],
        'fig:depthwise_top1': ['depthwise.csv','depthwise_figure.tex','depthwise_figure.svg'],
        'fig:method_overview': ['method_overview.html','method_overview.tex','method_panel_1.png','method_panel_2.png'],
        'tab:prompt_ablation': ['ablation.csv','ablation.tex'],
        'body:data': ['corpus_counts.csv','parent_counts.csv'],
        'body:statistics': ['main_paired_statistics.json','paired_tests.csv','text_comparisons.csv','ci_comparison.csv'],
        'body:coverage': ['computed_values.json','manuscript_claims.json','manuscript_coverage.md']}
    artifacts, absent = validate_artifact_files(output, planned)
    for label, detail in absent.items():
        missing['artifact:'+label] = detail['reason']
    conditions_complete = expected['mode'] == 'paper' and all(n in rows for n in CONDITIONS)
    artifacts_complete = not absent
    historical_match = not suite_dir and all(c.get('historical_value_verified') for c in claims if c['status'] != 'method_description')
    coverage = {'schema_version': 2,
        'paper_complete': conditions_complete and artifacts_complete and not missing,
        'conditions_complete': conditions_complete, 'report_artifacts_complete': artifacts_complete,
        'historical_numeric_match': historical_match,
        'completeness_definitions': {'conditions_complete': 'All seven structurally valid paper-population evaluations, not proof of historical values',
            'report_artifacts_complete': 'All required files generated and validated; tables may explicitly contain unverified NA values',
            'historical_numeric_match': 'All numeric claims match identified historical evidence at manuscript precision; corrected values are separate',
            'paper_complete': 'Valid consistent audits, both 10000-example training recipes, all seven paper-population conditions, all 159 fixed claims and required artifacts/computations; independent of historical numeric agreement'},
        'all_seven_outcomes_present': all(n in rows for n in CONDITIONS),
        'source_count': expected['count'], 'source_ids_sha256': expected['source_ids_sha256'],
        'missing': missing, 'claim_status_counts': claim_counts,
        'conditions': {n:{'model':v[0], 'prompt':v[1], 'outcomes_present': n in rows} for n,v in CONDITIONS.items()},
        'artifacts': artifacts, 'required_artifacts': planned, 'missing_artifacts': absent,
        'corrected_public_topk_available': bool(topk_corrected), 'provenance': provenance,
        'figure_asset_redistribution_ready': assets['redistribution_ready'],
        'publication_ready': False,
        'publication_note': 'Requires author review of corrections, figure permissions and exact artifact distribution; no external publication performed.',
        'inputs': {'tree_summary_sha256': file_sha256(tree_path), 'training_audit_sha256': file_sha256(audit_path),
                   'corpus_audit_sha256': file_sha256(corpus_path) if data else None,
                   'claim_manifest_sha256': file_sha256(manifest_path), 'claim_inventory_sha256': object_sha256(manuscript)},
        'statistics_note': 'Known recomputation: paired source bootstrap, not training-seed variation. Original-ID sorting also failed to reproduce all manuscript CI endpoints; historical CI procedure remains unidentified.'}
    write_json(output / 'coverage.json', coverage)
    if missing and not allow_partial:
        raise ValueError('Incomplete generated artifacts: ' + str(missing))
    return coverage



def merge_suites(directories, output, experiments="configs/experiments.json"):
    """CPU-only assembly for models served sequentially; original runs remain untouched."""
    import copy
    import shutil
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = load_experiments(experiments)
    merged, origins, all_rows = None, [], {}
    for directory in map(Path, directories):
        rows, expected, _, provenance = suite_rows(directory, config)
        plan = provenance["plan"]
        if merged is None:
            merged = copy.deepcopy(plan)
            merged["conditions"] = {}
        else:
            for field in ("profile", "mode", "tree_profile", "tree_identity", "inputs"):
                if merged[field] != plan[field]:
                    raise ValueError("Cannot merge suites with different " + field)
        for name in rows:
            if name in all_rows:
                raise ValueError("Duplicate condition while merging suites: " + name)
            all_rows[name] = rows[name]
            merged["conditions"][name] = plan["conditions"][name]
            origins.append((directory, name))
    if merged is None or not origins:
        raise ValueError("No validated condition outputs to merge")
    output.mkdir(parents=True)
    merged["assembled_from_independent_runs"] = True
    merged["output_dir"] = str(output)
    write_json(output / "suite.json", merged)
    write_json(output / "inputs.json", merged["inputs"])
    for directory, name in origins:
        target = output / name
        target.mkdir()
        for filename in ("results.jsonl", "summary.json", "invocation.json", "protocol.json", "inputs.json"):
            source = directory / name / filename
            shutil.copyfile(source, target / filename)
    # Validate cross-run model identity, source/extractor versions and every copied result.
    suite_rows(output, config)
    return {"conditions": list(all_rows), "all_seven_present": set(all_rows) == set(CONDITIONS)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--results-root", help="Optional external historical evidence directory")
    p.add_argument("--suite-dir")
    p.add_argument("--experiments", default="configs/experiments.json")
    p.add_argument("--tree-summary")
    p.add_argument("--training-audit")
    p.add_argument("--corpus-audit")
    p.add_argument("--project-root")
    p.add_argument("--claim-manifest")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--skip-statistics", action="store_true")
    p.add_argument("--bootstrap-iterations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    coverage = generate_report(a.output_dir, results_root=a.results_root, suite_dir=a.suite_dir,
        experiments=a.experiments, tree_summary=a.tree_summary, training_audit=a.training_audit, corpus_audit=a.corpus_audit, project_root=a.project_root, claim_manifest=a.claim_manifest,
        allow_partial=a.allow_partial, iterations=a.bootstrap_iterations, seed=a.seed,
        skip_statistics=a.skip_statistics)
    print(json.dumps({"paper_complete": coverage["paper_complete"], "missing": coverage["missing"]}, indent=2))


if __name__ == "__main__":
    main()
