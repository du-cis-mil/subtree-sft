"""Explicit seven-condition runner. Planning never starts servers or uses the network."""
import argparse
import copy
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

from .protocol import (CONDITIONS, PARENT_COUNTS, file_sha256, historical_tree_identity,
                       source_manifest, tree_identity, write_json)


def load_experiments(path):
    data = json.loads(Path(path).read_text())
    if set(data["conditions"]) != set(CONDITIONS):
        raise ValueError("Experiment manifest must declare all seven paper conditions")
    for name, (model, prompt) in CONDITIONS.items():
        item = data["conditions"][name]
        if (item["model"], item["prompt"]) != (model, prompt):
            raise ValueError(f"Incorrect model/prompt mapping: {name}")
    return data


def validate_models(mapping):
    models = mapping["models"]
    if set(models) != {"base", "public", "repro", "child"}:
        raise ValueError("Declare exactly the four paper models")
    checkpoints, names, urls = set(), set(), {}
    for key, model in models.items():
        checkpoint, name = model["checkpoint"], model["served_model_name"]
        if not checkpoint or not name or checkpoint in checkpoints or name in names:
            raise ValueError("Each model needs a distinct checkpoint and served model name")
        checkpoints.add(checkpoint)
        names.add(name)
        contexts = set()
        for server in model["deployments"]:
            context = server["context_length"]
            if type(context) is not int or context <= 0 or context in contexts or not server["urls"]:
                raise ValueError("Declare one deployment per positive context length")
            contexts.add(context)
            for url in server["urls"]:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
                    raise ValueError("Expected a credential-free HTTP endpoint")
                normalized = url.rstrip("/")
                if normalized in urls:
                    raise ValueError("An endpoint cannot stand for multiple models or context deployments")
                urls[normalized] = (key, context)
    return models


def make_plan(experiments, mapping, *, tree_dir, input_dir, output_dir, conditions=None,
              profile="historical", mode="paper", tree_profile="historical", python=sys.executable,
              fingerprints="configs/historical_tree_identity.json"):
    models = validate_models(mapping)
    if set(experiments["conditions"]) != set(CONDITIONS):
        raise ValueError("Missing paper condition")
    selected = list(CONDITIONS) if not conditions or conditions == ["all"] else conditions
    if len(selected) != len(set(selected)) or any(n not in CONDITIONS for n in selected):
        raise ValueError("Unknown/duplicate condition")
    if profile not in {"historical", "corrected-ranks"}:
        raise ValueError("Unknown profile")
    if mode not in {"paper", "toy"} or tree_profile not in {"historical", "rebuilt", "toy"}:
        raise ValueError("Unknown evaluation/tree mode")
    if (mode == "toy") != (tree_profile == "toy"):
        raise ValueError("Toy evaluation and toy tree profile must be selected together")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Use a new suite output directory: {output}")
    plan = {"schema_version": 1, "profile": profile, "mode": mode, "tree_profile": tree_profile,
            "tree_dir": str(tree_dir), "input_dir": str(input_dir), "output_dir": str(output),
            "artifact_fingerprints": str(fingerprints), "conditions": {},
            "model_identity_note": "Checkpoint/context are operator declarations; weight identity is unverified.",
            "input_inspection": "not performed in dry-run", "training": experiments["training"]}
    for name in selected:
        item = copy.deepcopy(experiments["conditions"][name])
        if (item["model"], item["prompt"]) != CONDITIONS[name]:
            raise ValueError(f"Incorrect model/prompt mapping: {name}")
        model = models[item["model"]]
        deployments = [d for d in model["deployments"] if d["context_length"] == item["context_length"]]
        if len(deployments) != 1:
            raise ValueError(f"No declared context deployment for {name}")
        spec = dict(experiments["evaluation"], **item)
        spec.update(condition_id=name, checkpoint=model["checkpoint"], model_name=model["served_model_name"],
                    urls=deployments[0]["urls"], weight_identity_verified=False,
                    format_error_ranks="legacy" if profile == "historical" else "none",
                    tree_dir=str(tree_dir), input_dir=str(input_dir), output_dir=str(output / name),
                    evaluation_mode=mode, tree_profile=tree_profile)
        command = [python, "-m", "soict_ir.evaluation", "--tree-dir", str(tree_dir),
                   "--input-dir", str(input_dir), "--output-dir", spec["output_dir"],
                   "--prompt", spec["prompt"], "--model-name", spec["model_name"],
                   "--ir-sglang-urls", *spec["urls"], "--evaluation-mode", mode,
                   "--tree-profile", tree_profile, "--artifact-fingerprints", str(fingerprints),
                   "--expected-manifest", str(output / "inputs.json"),
                   "--run-spec", str(output / "plans" / (name + ".json"))]
        for field in ("max_tokens", "top_logprobs", "api_max_retries", "format_max_retries", "workers",
                      "extractor_version", "format_error_ranks"):
            command.extend(["--" + field.replace("_", "-"), str(spec[field])])
        spec["command"] = command
        plan["conditions"][name] = spec
    return plan


def inspect_plan(plan):
    expected = source_manifest(plan["input_dir"], mode=plan["mode"])
    identity = tree_identity(plan["tree_dir"])
    if plan["tree_profile"] == "historical":
        fp = json.loads(Path(plan["artifact_fingerprints"]).read_text())
        if identity != historical_tree_identity(fp):
            raise ValueError("Historical tree identity mismatch")
    plan["input_inspection"] = "completed"
    plan["inputs"] = expected
    plan["tree_identity"] = identity
    for spec in plan["conditions"].values():
        spec["tree_identity"] = identity
        spec["source_ids_sha256"] = expected["source_ids_sha256"]
        spec["input_files_sha256"] = expected["input_files_sha256"]
    return plan


def validate_run_spec(spec, args, identity, expected):
    condition = spec["condition_id"]
    if condition not in CONDITIONS or (spec["model"], spec["prompt"]) != CONDITIONS[condition]:
        raise ValueError("Run condition model/prompt mismatch")
    for field in ("prompt", "model_name", "max_tokens", "top_logprobs", "api_max_retries",
                  "format_max_retries", "workers", "extractor_version", "format_error_ranks",
                  "tree_dir", "input_dir", "output_dir", "evaluation_mode", "tree_profile"):
        if getattr(args, field) != spec[field]:
            raise ValueError("Run spec differs from invocation: " + field)
    if args.ir_sglang_urls != spec["urls"] or identity != spec["tree_identity"]:
        raise ValueError("Run endpoint/tree identity mismatch")
    if (expected["source_ids_sha256"] != spec["source_ids_sha256"] or
            expected["input_files_sha256"] != spec["input_files_sha256"]):
        raise ValueError("Run source/input-byte identity mismatch")


def probe_server(url, model_name):
    """Only called by explicit --execute; checks the advertised name, never weight hashes."""
    from urllib.request import urlopen
    with urlopen(url.rstrip("/") + "/models", timeout=15) as response:
        names = [item["id"] for item in json.load(response)["data"]]
    if model_name not in names:
        raise ValueError(f"Server model mismatch: expected {model_name}, advertised {names}")
    return {"advertised_model_name": model_name, "weight_identity_verified": False,
            "context_length_verified": False}


def execute_plan(plan, *, extractor_factory=None, server_probe=probe_server):
    """Inject extractors/probes for network-free integration tests. Never launches a server."""
    from .evaluation import argument_parser, run_evaluation
    plan = inspect_plan(copy.deepcopy(plan))
    output = Path(plan["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    # All selected server identities are checked before the first inference request.
    probes = {}
    for spec in plan["conditions"].values():
        for url in spec["urls"]:
            key = (url, spec["model_name"])
            if key not in probes:
                probes[key] = server_probe(*key)
        spec["server_name_checks"] = [probes[(u, spec["model_name"])] for u in spec["urls"]]
    output.mkdir(parents=True)
    write_json(output / "inputs.json", plan["inputs"])
    write_json(output / "suite.json", plan)
    summaries = {}
    for name, spec in plan["conditions"].items():
        write_json(output / "plans" / (name + ".json"), spec)
        args = argument_parser().parse_args(spec["command"][3:])
        extractor = extractor_factory(spec) if extractor_factory else None
        summaries[name] = run_evaluation(args, extractor=extractor)
    write_json(output / "completion.json", {"conditions_completed": list(summaries),
        "all_seven_conditions_completed": set(summaries) == set(CONDITIONS),
        "paper_complete": plan["mode"] == "paper" and set(summaries) == set(CONDITIONS)})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="configs/experiments.json")
    parser.add_argument("--model-map", default="configs/models.example.json")
    parser.add_argument("--condition", nargs="+", choices=["all", *CONDITIONS], default=["all"])
    parser.add_argument("--tree-dir", default="data/trees/evaluation")
    parser.add_argument("--input-dir", default="data/eval")
    parser.add_argument("--output-dir")
    parser.add_argument("--profile", choices=["historical", "corrected-ranks"], default="historical")
    parser.add_argument("--evaluation-mode", choices=["paper", "toy"], default="paper")
    parser.add_argument("--tree-profile", choices=["historical", "rebuilt", "toy"], default="historical")
    parser.add_argument("--artifact-fingerprints", default="configs/historical_tree_identity.json")
    parser.add_argument("--inspect-inputs", action="store_true", help="Hash local inputs/trees on CPU")
    parser.add_argument("--plan-output", help="New JSON file; no servers or output run directories are created")
    parser.add_argument("--execute", action="store_true", help="Explicitly contact already-running model servers")
    args = parser.parse_args(argv)
    experiments = load_experiments(args.experiments)
    mapping = json.loads(Path(args.model_map).read_text())
    plan = make_plan(experiments, mapping, tree_dir=args.tree_dir, input_dir=args.input_dir,
                     output_dir=args.output_dir or "outputs/paper_" + args.profile,
                     conditions=args.condition, profile=args.profile, mode=args.evaluation_mode,
                     tree_profile=args.tree_profile, fingerprints=args.artifact_fingerprints)
    plan["experiments_sha256"] = file_sha256(args.experiments)
    plan["model_map_sha256"] = file_sha256(args.model_map)
    if args.inspect_inputs:
        inspect_plan(plan)
    if args.plan_output:
        write_json(args.plan_output, plan)
    if args.execute:
        execute_plan(plan)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
