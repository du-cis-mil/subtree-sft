import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from soict_ir.tree import HierarchicalSearchTree
from soict_ir.supervision import (build_dataset, build_title_index, resolve_target_ids,
    index_tree, nodes_for_targets, make_example)
from soict_ir.prompts import BRANCH_PROMPT, PAPER_PROMPT
from soict_ir.probability import IRProbabilityExtractor, SelectionResult, build_ir_prompt
from soict_ir.evaluation import LevelIREvaluator, summarize

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


toy = module("toy", ROOT / "examples/make_toy_data.py")
prefix = module("prefix", ROOT / "scripts/prepare_paper_selection.py")
splitter = module("splitter", ROOT / "scripts/split_eval.py")


class FakeExtractor:
    def __init__(self, fail=False, first=False):
        self.fail = fail
        self.first = first
        self.calls = []

    def get_selection_probabilities(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            return SelectionResult({"A": 0., "B": 0.}, None, -1, 2, format_error=True)
        first_title = kwargs["candidates"][0]["title"]
        chosen = "A" if self.first or first_title.endswith("1") else "B"
        probs = {label: .9 if label == chosen else .1 for label in ["A", "B"]}
        return SelectionResult(probs, chosen, "AB".index(chosen), 2)


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.tree_dir, self.records = toy.make(self.base / "toy")
        self.tree = HierarchicalSearchTree.load(str(self.tree_dir))
        self.targets, self.missing = resolve_target_ids(self.records[0], build_title_index(self.tree))

    def test_normalization_dedup_and_unmatched(self):
        self.assertEqual(self.targets, ["p3", "p1"])
        self.assertEqual([x["reason"] for x in self.missing], ["duplicate", "not_found_in_tree"])

    def test_internal_nodes_deduplicate_in_target_order(self):
        paths, _ = index_tree(self.tree.root)
        self.assertEqual([n["node_id"] for n in nodes_for_targets(self.targets, paths)], ["root", "right", "left"])

    def test_multi_positive_single_label_uses_first_target_not_first_child(self):
        _, leaves = index_tree(self.tree.root)
        row = make_example(self.tree, "toy", 42, self.records[0], "toy_1", self.tree.root,
                           self.targets, leaves, "first", BRANCH_PROMPT)
        self.assertEqual(row["metadata"]["positive_labels"], ["A", "B"])
        self.assertEqual(row["metadata"]["selected_label"], "B")

    def test_cap_after_expansion_partially_includes_final_source(self):
        summary = build_dataset(self.base / "toy/train.jsonl", self.tree_dir, self.base / "sft", 2)
        rows = [json.loads(x) for x in (self.base / "sft/train.jsonl").read_text().splitlines()]
        self.assertEqual(summary["unique_source_records"], 1)
        self.assertEqual([x["metadata"]["node_id"] for x in rows], ["root", "right"])

    def test_oversized_budget_is_incomplete_not_success(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_dataset(self.base / "toy/train.jsonl", self.tree_dir, self.base / "sft", 4)

    def test_existing_outputs_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            build_dataset(self.base / "toy/train.jsonl", self.tree_dir, self.base / "toy", 2)

    def test_candidate_order_and_medoid_title_are_preserved(self):
        candidates = self.tree.get_candidates_at_node(self.tree.root)
        self.assertEqual([x.paper_id for x in candidates], ["p1", "p3"])
        prompt, labels = build_ir_prompt("Q", "B", [{"title": x.title, "abstract": x.abstract} for x in candidates], prompt_name=PAPER_PROMPT)
        self.assertEqual(labels, {"A": "Synthetic paper 1", "B": "Synthetic paper 3"})
        self.assertLess(prompt.index("### Candidate [A]"), prompt.index("### Candidate [B]"))

    def test_source_record_or_aggregation_across_parents(self):
        evaluator = LevelIREvaluator(str(self.tree_dir), extractor=FakeExtractor())
        row = evaluator.evaluate_sample(str(self.base / "toy/eval/toy_1.json"))
        decisions = [d for d in row["decisions"] if d["level"] == 1]
        self.assertEqual([x["is_correct"] for x in decisions], [False, True])
        self.assertTrue(row["sample_level"]["1"]["is_correct_any"])
        self.assertEqual(row["sample_level"]["1"]["best_correct_rank"], 1)
        self.assertEqual(summarize([row])["sample_level_metrics"]["1"]["n"], 1)

    def test_wrong_ancestor_does_not_prevent_deeper_evaluation(self):
        record = dict(self.records[0], inspiration=[{"found_title": "Synthetic paper 3"}])
        path = self.base / "one.json"
        path.write_text(json.dumps(record))
        fake = FakeExtractor(first=True)
        row = LevelIREvaluator(str(self.tree_dir), extractor=fake).evaluate_sample(str(path))
        self.assertFalse(row["sample_level"]["0"]["is_correct_any"])
        self.assertTrue(row["sample_level"]["1"]["is_correct_any"])
        self.assertEqual(len(fake.calls), 2)

    def test_format_failure_has_no_rank_and_stays_in_denominator(self):
        row = LevelIREvaluator(str(self.tree_dir), extractor=FakeExtractor(fail=True), format_error_ranks="none").evaluate_sample(str(self.base / "toy/eval/toy_1.json"))
        self.assertIsNone(row["sample_level"]["0"]["best_correct_rank"])
        metrics = summarize([row], format_error_ranks="none")["sample_level_metrics"]["0"]
        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["topk_any_rate"]["top_5"], 0)

    def test_legacy_format_failure_ranks_preserve_historical_tie_order(self):
        row = LevelIREvaluator(str(self.tree_dir), extractor=FakeExtractor(fail=True)).evaluate_sample(str(self.base / "toy/eval/toy_1.json"))
        self.assertFalse(row["sample_level"]["0"]["is_correct_any"])
        self.assertEqual(row["sample_level"]["0"]["best_correct_rank"], 1)

    def test_prefix_is_original_bytes_without_shuffle(self):
        src = self.base / "original.jsonl"
        lines = [json.dumps({"conversations": [{"role": "user", "content": str(i)}]}) + "\n" for i in range(3)]
        src.write_text("".join(lines))
        prefix.prepare(src, self.base / "prefix", 2)
        self.assertEqual((self.base / "prefix/train.jsonl").read_bytes(), "".join(lines[:2]).encode())

    def test_eval_split_rejects_unsafe_and_duplicate_ids(self):
        src = self.base / "unsafe.jsonl"
        src.write_text(json.dumps({"source_id": "../escape"}) + "\n")
        with self.assertRaises(ValueError):
            splitter.split(src, self.base / "unsafe")
        src.write_text('{"source_id":"same"}\n{"source_id":"same"}\n')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            splitter.split(src, self.base / "duplicate")


class ProbabilityTest(unittest.TestCase):
    def test_scores_are_read_at_selection_marker_not_reasoning(self):
        extractor = IRProbabilityExtractor.__new__(IRProbabilityExtractor)
        text = "reason Candidate [A] then **Selected ID starts:** [B]"
        tokens = [SimpleNamespace(token=text[:-2], logprob=-9., top_logprobs=[]),
                  SimpleNamespace(token="B", logprob=-.2, top_logprobs=[
                      SimpleNamespace(token="A", logprob=-2.), SimpleNamespace(token="B", logprob=-.2)]),
                  SimpleNamespace(token="]", logprob=0., top_logprobs=[])]
        result = extractor._extract_label_logprobs(text, SimpleNamespace(content=tokens), ["A", "B", "C"])
        self.assertEqual(result, {"A": -2., "B": -.2, "C": -100.})
        probabilities = extractor._logprobs_to_probs(result, 1.)
        self.assertAlmostEqual(sum(probabilities.values()), 1.)
        self.assertEqual(max(probabilities, key=probabilities.get), "B")

    def test_truncated_output_has_no_selection(self):
        extractor = IRProbabilityExtractor.__new__(IRProbabilityExtractor)
        self.assertIsNone(extractor._extract_label_logprobs("no marker", SimpleNamespace(content=[]), ["A", "B"]))

    def test_missing_logprobs_is_an_error(self):
        with self.assertRaises(RuntimeError):
            IRProbabilityExtractor.__new__(IRProbabilityExtractor)._extract_label_logprobs("x", None, ["A", "B"])

    def test_alternate_marker_uses_its_actual_offset(self):
        text = "Selected ID starts:** [B]"
        tokens = [SimpleNamespace(token=text[:-2], logprob=-9., top_logprobs=[]),
                  SimpleNamespace(token="B", logprob=-.3, top_logprobs=[]),
                  SimpleNamespace(token="]", logprob=0., top_logprobs=[])]
        result = IRProbabilityExtractor.__new__(IRProbabilityExtractor)._extract_label_logprobs(text, SimpleNamespace(content=tokens), ["A", "B"])
        self.assertEqual(result["B"], -.3)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "analysis extra is optional")
class StatisticsTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        from soict_ir.statistics import exact_mcnemar_p
        self.assertEqual(exact_mcnemar_p([True]*4, [False]*4), (4, 0, .125))
        self.assertEqual(exact_mcnemar_p([True], [True]), (0, 0, 1.))

    def test_pairing_must_not_drop_sources(self):
        analyzer = module("analyzer", ROOT / "scripts/analyze_results.py")
        with self.assertRaisesRegex(ValueError, "identical"):
            analyzer.compare({"a": [True]*5}, {"b": [True]*5})


class ArtifactTest(unittest.TestCase):
    def test_release_ignores_runtime_outputs_and_parent_history(self):
        release = module("release", ROOT / "scripts/make_release.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in release.TOP_FILES:
                (root / name).write_text("test\n")
            (root / "outputs").mkdir()
            (root / "outputs/private.txt").write_text("not for release")
            (root / ".git").mkdir()
            (root / ".git/config").write_text("parent remote")
            self.assertEqual({p.name for p in release.release_files(root)}, release.TOP_FILES)

    def test_source_audit_matches_context_and_preserves_source_counts(self):
        audit = module("audit", ROOT / "scripts/audit_training_data.py")
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            record = {"source_id": "one", "research_question": "Q", "background_survey": "B"}
            (base / "source.jsonl").write_text(json.dumps(record) + "\n")
            prompt, _ = build_ir_prompt("Q", "B", [{"title": "a", "abstract": "a"}, {"title": "b", "abstract": "b"}], prompt_name=PAPER_PROMPT)
            (base / "paper.jsonl").write_text(json.dumps({"conversations": [{"role": "user", "content": prompt}]}) + "\n")
            (base / "child.jsonl").write_text(json.dumps({"metadata": {"source_id": "one", "node_level": 0}}) + "\n")
            result = audit.audit(base / "source.jsonl", base / "paper.jsonl", base / "child.jsonl")
            self.assertEqual(result["shared_sources"], 1)
            self.assertEqual(result["paper_selection_unique_sources"], 1)

    def test_granularity_uses_subtree_paper_counts(self):
        summary = module("granularity", ROOT / "scripts/summarize_tree.py")
        with tempfile.TemporaryDirectory() as temp:
            tree_dir, _ = toy.make(Path(temp) / "toy")
            result = summary.granularity(HierarchicalSearchTree.load(str(tree_dir)))
            self.assertEqual(result["num_papers"], 4)
            self.assertEqual(result["depths"]["0"]["mean_papers_per_candidate"], 2)
            self.assertEqual(result["depths"]["1"]["median_papers_per_candidate"], 1)


if __name__ == "__main__":
    unittest.main()
