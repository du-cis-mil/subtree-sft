import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from soict_ir.protocol import (CONDITIONS, expected_sources, source_manifest, tree_identity,
                               validate_rows, write_json, file_sha256)
from soict_ir.suite import (load_experiments, make_plan, inspect_plan, execute_plan,
                            validate_models, probe_server)
from soict_ir.probability import IRProbabilityExtractor, SelectionResult
from soict_ir.metrics import source_metrics
from soict_ir.reporting import generate_report, suite_rows, merge_suites
from soict_ir.data_audit import audit as corpus_audit

ROOT = Path(__file__).resolve().parents[1]


class FakeExtractor:
    def __init__(self, calls, spec, error=False):
        self.calls, self.spec, self.error = calls, spec, error

    def get_selection_probabilities(self, **kwargs):
        self.calls.append((self.spec["condition_id"], kwargs))
        if self.error:
            raise RuntimeError("synthetic API failure")
        return SelectionResult({"A": .9, "B": .1}, "A", 0, 2)


def fixture(directory):
    tree, inputs = directory / "tree", directory / "inputs"
    tree.mkdir(); inputs.mkdir()

    def leaf(pid, depth):
        return {"node_id": pid, "level": depth, "paper_id": pid, "centroid_paper_id": pid,
                "num_papers": 1, "is_leaf": True, "children": []}

    node = leaf("p0", 5)
    for d in reversed(range(5)):
        node = {"node_id": "n" + str(d), "level": d, "centroid_paper_id": "p0",
                "num_papers": 6-d, "is_leaf": False, "children": [node, leaf("p" + str(d+1), d+1)]}
    papers = [{"paper_id": "p" + str(i), "title": "Synthetic " + str(i), "abstract": "Invented."} for i in range(6)]
    for filename, data in [("hierarchical_tree.json", node), ("papers.json", papers), ("tree_config.json", {"seed": 42, "embedding_dim": 768, "branching_factor": 15, "max_depth": 5, "use_medoid": 1})]:
        write_json(tree / filename, data)
    for i in range(2):
        sid = "toy_" + str(i)
        write_json(inputs / (sid + ".json"), {"source_id": sid, "research_question": "Synthetic question",
                   "background_survey": "Synthetic background", "inspiration": [{"found_title": "Synthetic 0"}]})
    return tree, inputs


def report_audits(tree, inputs, base, examples=10000):
    """Synthetic data audits: no manuscript values or real model outcomes."""
    train, test = base / "train.jsonl", base / "test.jsonl"
    papers = json.loads((tree / "papers.json").read_text())
    train.write_text("\n".join(json.dumps(dict(p, source_id="train_"+str(i))) for i,p in enumerate(papers))+"\n")
    test.write_text("\n".join(json.dumps(json.loads(p.read_text())) for p in sorted(inputs.glob("*.json")))+"\n")
    data = corpus_audit(train, test, tree)
    training = {"paper_selection_examples": examples, "child_subtree_examples": examples,
        "paper_selection_unique_sources": 3, "child_subtree_unique_sources": 2, "shared_sources": 1,
        "child_depth_distribution": {str(d):examples//5 for d in range(5)}, "schema_version": 1,
        "provenance": {"kind": "synthetic_fixture", "inputs": {"source_train": data["inputs"]["source_train_sha256"],
            "paper_training_prefix": "0"*64, "child_training_prefix": "1"*64}, "weight_identity_verified": False}}
    for name, value in [("granularity.json", data["granularity"]), ("corpus.json", data), ("training.json", training)]:
        write_json(base / name, value)
    return dict(tree_summary=base/"granularity.json", corpus_audit=base/"corpus.json", training_audit=base/"training.json")


class SuiteTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.tree, self.inputs = fixture(self.base)
        self.experiments = load_experiments(ROOT / "configs/experiments.json")
        self.mapping = json.loads((ROOT / "configs/models.example.json").read_text())

    def plan(self, **kwargs):
        return make_plan(self.experiments, self.mapping, tree_dir=self.tree, input_dir=self.inputs,
                         output_dir=self.base / "run", mode="toy", tree_profile="toy", **kwargs)

    def execute(self, plan=None, error=False):
        calls = []
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
             patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
            result = execute_plan(plan or self.plan(),
                extractor_factory=lambda s: FakeExtractor(calls, s, error),
                server_probe=lambda url, name: {"advertised_model_name": name, "weight_identity_verified": False})
        return result, calls

    def test_all_seven_conditions_and_repro_child_selection(self):
        plan = self.plan()
        self.assertEqual(set(plan["conditions"]), set(CONDITIONS))
        child = plan["conditions"]["repro_child"]
        self.assertEqual((child["model"], child["prompt"], child["max_tokens"]), ("repro", "child-subtree", 2048))
        self.assertIn("unverified", child["settings_evidence"])
        self.assertEqual(set(self.plan(conditions=["repro_child"])["conditions"]), {"repro_child"})
        self.assertEqual(plan["conditions"]["base_child"]["max_tokens"], 512)
        self.assertEqual(plan["conditions"]["public_original"]["context_length"], 32768)
        self.assertIsNone(plan["conditions"]["base_child"]["historical_context_length"])

    def test_seven_condition_pipeline_uses_no_network_and_correct_prompts(self):
        result, calls = self.execute()
        self.assertEqual(len(result), 7)
        self.assertEqual(len(calls), 7*2*5)
        for name, kwargs in calls:
            self.assertEqual(kwargs["max_tokens"], 512 if name == "base_child" else 2048)
            self.assertEqual(kwargs["api_max_retries"], 10)
            self.assertEqual(kwargs["format_max_retries"], 1)
            self.assertEqual("branch_retrieval" in kwargs["prompt_name"], CONDITIONS[name][1] == "child-subtree")
        self.assertFalse(any(s["paper_complete"] for s in result.values()))
        self.assertTrue(json.loads((self.base / "run/completion.json").read_text())["all_seven_conditions_completed"])

    def test_full_report_rejects_toy_and_partial_report_reaches_tables(self):
        self.execute()
        audits = report_audits(self.tree, self.inputs, self.base)
        kwargs = dict(suite_dir=self.base / "run", experiments=ROOT / "configs/experiments.json",
                      **audits, skip_statistics=True)
        with self.assertRaisesRegex(ValueError, "Incomplete paper report"):
            generate_report(self.base / "full", **kwargs)
        coverage = generate_report(self.base / "partial", allow_partial=True, **kwargs)
        self.assertFalse(coverage["paper_complete"])
        self.assertTrue((self.base / "partial/ablation.csv").exists())
        self.assertTrue((self.base / "partial/depthwise_figure.tex").exists())

    def test_missing_condition_is_reported_not_filled_from_manuscript(self):
        self.execute()
        (self.base / "run/repro_child/results.jsonl").unlink()
        rows, _, missing, _ = suite_rows(self.base / "run", self.experiments)
        self.assertNotIn("repro_child", rows)
        self.assertIn("repro_child", missing)

    def test_sequential_model_runs_merge_without_overwriting_originals(self):
        first = self.plan(conditions=["repro_child", "repro_original"])
        self.execute(first)
        second = make_plan(
            self.experiments, self.mapping, tree_dir=self.tree, input_dir=self.inputs,
            output_dir=self.base / "second", mode="toy", tree_profile="toy", conditions=["base_child"])
        self.execute(second)
        original_hash = file_sha256(self.base / "run/repro_child/results.jsonl")
        merged = merge_suites([self.base / "run", self.base / "second"], self.base / "merged",
                              ROOT / "configs/experiments.json")
        self.assertEqual(set(merged["conditions"]), {"repro_child", "repro_original", "base_child"})
        self.assertEqual(file_sha256(self.base / "run/repro_child/results.jsonl"), original_hash)
        self.assertEqual(file_sha256(self.base / "merged/repro_child/results.jsonl"), original_hash)
        with self.assertRaisesRegex(ValueError, "Duplicate condition"):
            merge_suites([self.base / "run", self.base / "run"], self.base / "duplicate",
                         ROOT / "configs/experiments.json")

    def test_changed_model_declaration_is_rejected(self):
        bad = copy.deepcopy(self.mapping)
        bad["models"]["repro"]["checkpoint"] = bad["models"]["child"]["checkpoint"]
        with self.assertRaisesRegex(ValueError, "distinct checkpoint"):
            validate_models(bad)
        bad = copy.deepcopy(self.mapping)
        bad["models"]["repro"]["deployments"][0]["urls"] = bad["models"]["child"]["deployments"][0]["urls"]
        with self.assertRaisesRegex(ValueError, "endpoint"):
            validate_models(bad)

    def test_wrong_model_prompt_and_context_mapping_are_rejected(self):
        self.experiments["conditions"]["repro_child"]["model"] = "child"
        with self.assertRaisesRegex(ValueError, "model/prompt"):
            self.plan()
        self.experiments["conditions"]["repro_child"]["model"] = "repro"
        self.mapping["models"]["public"]["deployments"] = self.mapping["models"]["public"]["deployments"][:1]
        with self.assertRaisesRegex(ValueError, "context deployment"):
            self.plan()

    def test_advertised_server_model_mismatch_is_rejected(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"data":[{"id":"wrong-model"}]}'
        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "Server model mismatch"):
                probe_server("http://example.invalid/v1", "soict-repro")

    def test_existing_output_rejected_and_api_errors_preserved(self):
        with self.assertRaisesRegex(ValueError, "Incomplete evaluation"):
            self.execute(error=True)
        summary = json.loads((self.base / "run/base_child/summary.json").read_text())
        self.assertFalse(summary["paper_complete"])
        self.assertEqual(summary["statuses"], {"error": 2})
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_mutated_tree_protocol_rejected_during_reporting(self):
        self.execute()
        p = self.base / "run/repro_child/protocol.json"
        data = json.loads(p.read_text()); data["tree_identity"]["structure"] = "wrong"
        p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "tree identity"):
            suite_rows(self.base / "run", self.experiments)

    def test_inputs_are_frozen_and_filename_id_mismatch_rejected(self):
        a = source_manifest(self.inputs, mode="toy")
        p = self.inputs / "toy_0.json"
        data = json.loads(p.read_text());data["research_question"] += " changed"
        p.write_text(json.dumps(data))
        b = source_manifest(self.inputs, mode="toy")
        self.assertEqual(a["source_ids_sha256"], b["source_ids_sha256"])
        self.assertNotEqual(a["input_files_sha256"], b["input_files_sha256"])
        data["source_id"] = "wrong";p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "filename"):
            source_manifest(self.inputs, mode="toy")

    def test_incomplete_population_fails_before_extractor_construction(self):
        from soict_ir.evaluation import argument_parser, run_evaluation
        args = argument_parser().parse_args(["--tree-dir", str(self.tree), "--input-dir", str(self.inputs),
                                            "--output-dir", str(self.base / "paper")])
        with patch("soict_ir.probability.IRProbabilityExtractor", side_effect=AssertionError("No model call allowed")):
            with self.assertRaisesRegex(ValueError, "1658"):
                run_evaluation(args)


class CompletenessTest(unittest.TestCase):
    def test_wrong_tree_and_parent_counts_are_rejected(self):
        expected = expected_sources(["a"], mode="toy")
        row = {"source_id": "a", "correct": [True]*5,
               "decisions": [{"node_id": str(d), "level": d, "is_correct": True, "format_error": False} for d in range(5)]}
        with self.assertRaisesRegex(ValueError, "tree identity"):
            validate_rows([row], expected, expected_tree={"hash": "a"}, actual_tree={"hash": "b"})
        with self.assertRaisesRegex(ValueError, "Parent counts"):
            validate_rows([row], expected, parent_counts=[2]*5)
    def test_53_records_cannot_be_a_complete_paper_evaluation(self):
        with self.assertRaisesRegex(ValueError, "1658"):
            expected_sources([str(i) for i in range(53)])

    def test_equal_count_different_ids_duplicates_missing_depth_and_error_rejected(self):
        expected = expected_sources(["a", "b"], mode="toy")
        valid = [{"source_id": s, "correct": [True]*5} for s in ["a", "b"]]
        validate_rows(valid, expected)
        for rows in [[valid[0], valid[0]], [valid[0], dict(valid[1], source_id="c")],
                     [valid[0], dict(valid[1], correct=[True]*4)],
                     [valid[0], dict(valid[1], status="error")]]:
            with self.assertRaises(ValueError): validate_rows(rows, expected)

    def test_format_failure_with_another_valid_parent_uses_valid_best_rank(self):
        ds = [{"node_id": str(i), "level": d, "candidate_count": 5, "is_correct": False,
               "format_error": i == 0, "best_correct_rank": 1 if i == 0 else 3}
              for d in range(5) for i in range(2)]
        row = {"source_id": "a", "correct": [False]*5, "decisions": ds}
        metrics = source_metrics([row], "none")["sample_level_metrics"]["0"]
        self.assertEqual(metrics["mean_valid_rank"], 3)
        self.assertEqual(metrics["topk_any_rate"]["top_2"], 0)
        self.assertEqual(metrics["topk_any_rate"]["top_3"], 1)
        for d in ds:d["format_error"] = True
        metrics = source_metrics([row], "none")["sample_level_metrics"]["0"]
        self.assertEqual(metrics["topk_any_rate"]["top_5"], 0)
        self.assertEqual(metrics["valid_rank_sources"], 0)
        self.assertIsNone(metrics["mean_valid_rank"])

    def test_compact_outcomes_do_not_invent_format_error_evidence(self):
        with self.assertRaisesRegex(ValueError, "decision-level"):
            source_metrics([{"source_id": "a", "correct": [False]*5, "best_rank": [1]*5}], "none")


class MarkerCompatibilityTest(unittest.TestCase):
    def extract(self, text, version):
        extractor = IRProbabilityExtractor.__new__(IRProbabilityExtractor)
        extractor.extractor_version = version
        tokens = [SimpleNamespace(token=text[:-2], logprob=-9., top_logprobs=[]),
                  SimpleNamespace(token="B", logprob=-.3, top_logprobs=[]),
                  SimpleNamespace(token="]", logprob=0., top_logprobs=[])]
        return extractor._extract_label_logprobs(text, SimpleNamespace(content=tokens), ["A", "B"])

    def test_standard_marker_is_identical_in_both_modes(self):
        text = "Reasoning. **Selected ID starts:** [B]"
        self.assertEqual(self.extract(text, "legacy"), self.extract(text, "fixed"))

    def test_alternate_marker_legacy_offset_is_explicit(self):
        text = "Selected ID starts:** [B]"
        self.assertIsNone(self.extract(text, "legacy"))
        self.assertEqual(self.extract(text, "fixed")["B"], -.3)


if __name__ == '__main__': unittest.main()
