# Adapted for the standalone SoICT release (2026): local imports and CLI paths revised.
# MOOSE-Star-derived code; see LICENSE and NOTICE.
import argparse, collections, json, os, statistics, time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from .tree import HierarchicalSearchTree
from .supervision import normalize_title
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class LevelIREvaluator:
    """Independent ground-truth-path decisions with fixed medoid representatives."""
    def __init__(self, tree_dir, sglang_urls=None, *, extractor=None,
                 prompt_name="inspiration_branch_retrieval_with_reasoning_with_alphabetical_candidates",
                 max_tokens=2048, top_logprobs=30, api_max_retries=1, format_max_retries=1,
                 format_error_ranks="legacy", model_name="default", extractor_version="fixed"):
        self.tree = HierarchicalSearchTree.load(tree_dir)
        if extractor is None:
            from .probability import IRProbabilityExtractor
            extractor = IRProbabilityExtractor(base_urls=sglang_urls or ["http://127.0.0.1:30000/v1"],
                                               model_name=model_name, extractor_version=extractor_version)
        self.extractor = extractor
        self.softmax_temperature = 1.0
        self.generation_temperature = 0.0
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.prompt_name = prompt_name
        self.api_max_retries = api_max_retries
        self.format_max_retries = format_max_retries
        self.representative_mode = "medoid"
        if format_error_ranks not in {"legacy", "none"}:
            raise ValueError("format_error_ranks must be legacy or none")
        self.format_error_ranks = format_error_ranks
        self.title_to_paper_id = {normalize_title(p.get("title", "")): pid for pid, p in self.tree.papers.items()}
        self.paper_to_path = {}
        self.node_to_leaf_papers = {}
        self.paper_node_to_child = collections.defaultdict(dict)
        self._index_tree(self.tree.root, [], [])

    def get_representative_candidates(self, node, data, **kwargs):
        candidates = self.tree.get_candidates_at_node(node)
        return ([{"title": c.title, "abstract": c.abstract} for c in candidates],
                [{"child_index": c.index, "child_node_id": c.node_id,
                  "representative_paper_id": c.paper_id, "representative_source": "medoid",
                  "representative_score": None} for c in candidates])

    def _index_tree(
        self,
        node: Dict,
        path: List[Dict],
        ancestor_child_pairs: List[Tuple[str, int]],
    ) -> Set[str]:
        current_path = path + [node]
        if node.get("is_leaf"):
            paper_id = node["paper_id"]
            self.paper_to_path[paper_id] = current_path
            self.paper_node_to_child[paper_id] = dict(ancestor_child_pairs)
            leaves = {paper_id}
        else:
            leaves = set()
            for child_idx, child in enumerate(node.get("children", [])):
                child_pairs = ancestor_child_pairs + [(node["node_id"], child_idx)]
                leaves.update(self._index_tree(child, current_path, child_pairs))
        self.node_to_leaf_papers[node["node_id"]] = leaves
        return leaves

    def resolve_targets(self, data: Dict) -> Tuple[List[str], List[Dict]]:
        target_ids = []
        missing = []
        for insp in data.get("inspiration", []):
            title = insp.get("found_title") or insp.get("supposed_title") or ""
            paper_id = self.title_to_paper_id.get(normalize_title(title))
            if paper_id and paper_id not in target_ids:
                target_ids.append(paper_id)
            else:
                missing.append({
                    "title": title,
                    "reason": "not_found_in_tree" if not paper_id else "duplicate",
                })
        return target_ids, missing

    def target_child_indices(self, node: Dict, target_ids: Set[str]) -> List[int]:
        indices = []
        for idx, child in enumerate(node.get("children", [])):
            child_leaves = self.node_to_leaf_papers.get(child["node_id"], set())
            if child_leaves.intersection(target_ids):
                indices.append(idx)
        return indices

    def target_ids_in_child(self, child: Dict, target_ids: Set[str]) -> List[str]:
        child_leaves = self.node_to_leaf_papers.get(child["node_id"], set())
        return sorted(child_leaves.intersection(target_ids))

    def nodes_by_level_for_targets(self, target_ids: List[str]) -> Dict[int, Dict[str, Dict]]:
        by_level: Dict[int, Dict[str, Dict]] = collections.defaultdict(dict)
        for paper_id in target_ids:
            for node in self.paper_to_path.get(paper_id, []):
                if node.get("is_leaf") or len(node.get("children", [])) <= 1:
                    continue
                by_level[node.get("level", 0)][node["node_id"]] = node
        return by_level

    def evaluate_decision(
        self,
        data: Dict,
        source_id: str,
        node: Dict,
        target_ids: Set[str],
        query_scores: Optional[Dict[str, float]] = None,
        sorted_query_scores: Optional[List[Tuple[str, float]]] = None,
    ) -> Dict:
        candidates = self.tree.get_candidates_at_node(node)
        ir_candidates, representative_metadata = self.get_representative_candidates(
            node,
            data,
            query_scores=query_scores,
            sorted_query_scores=sorted_query_scores,
        )
        correct_indices = self.target_child_indices(node, target_ids)
        if not correct_indices:
            raise RuntimeError(f"No target child found at node {node['node_id']}.")

        result = self.extractor.get_selection_probabilities(
            research_question=data["research_question"],
            background_survey=data["background_survey"],
            candidates=ir_candidates,
            previous_hypothesis=None,
            softmax_temperature=self.softmax_temperature,
            generation_temperature=self.generation_temperature,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs,
            return_response=False,
            prompt_name=self.prompt_name,
            api_max_retries=self.api_max_retries,
            format_max_retries=self.format_max_retries,
        )

        valid_labels = LABELS[:len(candidates)]
        sorted_labels = sorted(valid_labels, key=lambda label: result.probabilities.get(label, 0.0), reverse=True)
        correct_labels = [LABELS[idx] for idx in correct_indices]
        correct_ranks = [sorted_labels.index(label) + 1 for label in correct_labels if label in sorted_labels]
        keep_rank = not result.format_error or self.format_error_ranks == "legacy"
        best_rank = min(correct_ranks) if correct_ranks and keep_rank else None
        selected_idx = result.selected_index if 0 <= result.selected_index < len(candidates) else None
        selected_label = LABELS[selected_idx] if selected_idx is not None else None
        selected_prob = result.probabilities.get(selected_label, 0.0) if selected_label else 0.0
        is_correct = selected_idx in correct_indices if selected_idx is not None else False

        return {
            "source_id": source_id,
            "node_id": node["node_id"],
            "level": node.get("level", 0),
            "candidate_count": len(candidates),
            "target_count": len(target_ids),
            "correct_indices": correct_indices,
            "correct_labels": correct_labels,
            "correct_child_targets": {
                LABELS[idx]: self.target_ids_in_child(node["children"][idx], target_ids)
                for idx in correct_indices
            },
            "representative_mode": self.representative_mode,
            "representatives": representative_metadata,
            "correct_branch_representative_is_target": {
                LABELS[idx]: representative_metadata[idx]["representative_paper_id"] in target_ids
                for idx in correct_indices
            },
            "selected_index": selected_idx,
            "selected_label": selected_label,
            "selected_prob": selected_prob,
            "is_correct": is_correct,
            "format_error": result.format_error,
            "best_correct_rank": best_rank,
            "best_correct_prob": max((result.probabilities.get(label, 0.0) for label in correct_labels), default=0.0),
            "prob_mass_on_correct": sum(result.probabilities.get(label, 0.0) for label in correct_labels),
            "branch_evaluations": [
                {
                    "source_id": source_id,
                    "node_id": node["node_id"],
                    "level": node.get("level", 0),
                    "candidate_count": len(candidates),
                    "target_count": len(target_ids),
                    "correct_index": idx,
                    "correct_label": LABELS[idx],
                    "correct_prob": result.probabilities.get(LABELS[idx], 0.0),
                    "correct_rank": sorted_labels.index(LABELS[idx]) + 1 if LABELS[idx] in sorted_labels and keep_rank else None,
                    "target_paper_ids": self.target_ids_in_child(node["children"][idx], target_ids),
                    "representative_paper_id": representative_metadata[idx]["representative_paper_id"],
                    "representative_is_target": representative_metadata[idx]["representative_paper_id"] in target_ids,
                    "selected_index": selected_idx,
                    "selected_label": selected_label,
                    "selected_prob": selected_prob,
                    "is_correct": selected_idx == idx,
                    "format_error": result.format_error,
                }
                for idx in correct_indices
            ],
            "ranked_labels": [
                {
                    "rank": rank + 1,
                    "label": label,
                    "prob": result.probabilities.get(label, 0.0),
                    "title": ir_candidates[LABELS.index(label)]["title"],
                    "representative_paper_id": representative_metadata[LABELS.index(label)]["representative_paper_id"],
                    "contains_any_target": label in correct_labels,
                }
                for rank, label in enumerate(sorted_labels)
            ],
        }

    def evaluate_sample(self, path: str) -> Dict:
        started = time.time()
        source_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)

        target_ids, missing = self.resolve_targets(data)
        sample_result = {
            "source_id": data.get("source_id") or source_id,
            "input_file": path,
            "target_paper_ids": target_ids,
            "missing_targets": missing,
            "target_count": len(target_ids),
            "decisions": [],
            "branch_decisions": [],
            "sample_level": {},
            "status": "ok",
            "elapsed_seconds": None,
        }

        if not target_ids:
            sample_result["status"] = "no_targets"
            sample_result["elapsed_seconds"] = time.time() - started
            return sample_result

        target_set = set(target_ids)
        nodes_by_level = self.nodes_by_level_for_targets(target_ids)
        query_scores = None
        sorted_query_scores = None
        for level in sorted(nodes_by_level):
            level_decisions = []
            for node in nodes_by_level[level].values():
                level_decisions.append(
                    self.evaluate_decision(
                        data,
                        source_id,
                        node,
                        target_set,
                        query_scores=query_scores,
                        sorted_query_scores=sorted_query_scores,
                    )
                )
            sample_result["decisions"].extend(level_decisions)
            for decision in level_decisions:
                sample_result["branch_decisions"].extend(decision["branch_evaluations"])
            sample_result["sample_level"][str(level)] = {
                "num_decisions": len(level_decisions),
                "num_target_branches": sum(len(d["branch_evaluations"]) for d in level_decisions),
                "is_correct_any": any(d["is_correct"] for d in level_decisions),
                "best_correct_rank": min(
                    (d["best_correct_rank"] for d in level_decisions if d["best_correct_rank"] is not None),
                    default=None,
                ),
                "prob_mass_on_correct_max": max((d["prob_mass_on_correct"] for d in level_decisions), default=0.0),
            }

        sample_result["elapsed_seconds"] = time.time() - started
        return sample_result

def summarize(results: List[Dict], format_error_ranks="legacy") -> Dict:
    decision_by_level = collections.defaultdict(list)
    branch_by_level = collections.defaultdict(list)
    sample_by_level = collections.defaultdict(list)
    statuses = collections.Counter()
    target_counts = collections.Counter()

    for result in results:
        statuses[result.get("status", "unknown")] += 1
        target_counts[result.get("target_count", 0)] += 1
        for decision in result.get("decisions", []):
            decision_by_level[str(decision["level"])].append(decision)
        for branch_decision in result.get("branch_decisions", []):
            branch_by_level[str(branch_decision["level"])].append(branch_decision)
        for level, level_result in result.get("sample_level", {}).items():
            sample_by_level[level].append(level_result)

    def decision_metrics(decisions: List[Dict]) -> Dict:
        ranks = [d["best_correct_rank"] for d in decisions if d.get("best_correct_rank") is not None]
        denominator = len(ranks) if format_error_ranks == "legacy" else len(decisions)
        correct = sum(1 for d in decisions if d.get("is_correct"))
        format_errors = sum(1 for d in decisions if d.get("format_error"))
        return {
            "n": len(decisions),
            "correct": correct,
            "format_errors": format_errors,
            "format_error_rate": format_errors / len(decisions) if decisions else None,
            "accuracy": correct / len(decisions) if decisions else None,
            "best_rank_mean": statistics.mean(ranks) if ranks else None,
            "best_rank_median": statistics.median(ranks) if ranks else None,
            "topk": {
                f"top_{k}": sum(1 for r in ranks if r <= k)
                for k in [1, 2, 3, 5, 10, 15]
            },
            "topk_rate": {
                f"top_{k}": (sum(1 for r in ranks if r <= k) / denominator if denominator else None)
                for k in [1, 2, 3, 5, 10, 15]
            },
        }

    def branch_metrics(branches: List[Dict]) -> Dict:
        ranks = [b["correct_rank"] for b in branches if b.get("correct_rank") is not None]
        denominator = len(ranks) if format_error_ranks == "legacy" else len(branches)
        correct = sum(1 for b in branches if b.get("is_correct"))
        format_errors = sum(1 for b in branches if b.get("format_error"))
        return {
            "n": len(branches),
            "correct": correct,
            "format_errors": format_errors,
            "format_error_rate": format_errors / len(branches) if branches else None,
            "accuracy": correct / len(branches) if branches else None,
            "correct_rank_mean": statistics.mean(ranks) if ranks else None,
            "correct_rank_median": statistics.median(ranks) if ranks else None,
            "topk": {
                f"top_{k}": sum(1 for r in ranks if r <= k)
                for k in [1, 2, 3, 5, 10, 15]
            },
            "topk_rate": {
                f"top_{k}": (sum(1 for r in ranks if r <= k) / denominator if denominator else None)
                for k in [1, 2, 3, 5, 10, 15]
            },
        }

    def sample_metrics(samples: List[Dict]) -> Dict:
        ranks = [s["best_correct_rank"] for s in samples if s.get("best_correct_rank") is not None]
        denominator = len(ranks) if format_error_ranks == "legacy" else len(samples)
        correct = sum(1 for s in samples if s.get("is_correct_any"))
        return {
            "n": len(samples),
            "correct_any": correct,
            "accuracy_any": correct / len(samples) if samples else None,
            "best_rank_mean": statistics.mean(ranks) if ranks else None,
            "best_rank_median": statistics.median(ranks) if ranks else None,
            "topk_any": {
                f"top_{k}": sum(1 for r in ranks if r <= k)
                for k in [1, 2, 3, 5, 10, 15]
            },
            "topk_any_rate": {
                f"top_{k}": (sum(1 for r in ranks if r <= k) / denominator if denominator else None)
                for k in [1, 2, 3, 5, 10, 15]
            },
        }

    return {
        "num_samples": len(results),
        "statuses": dict(statuses),
        "target_count_distribution": dict(sorted(target_counts.items())),
        "decision_level_metrics": {
            level: decision_metrics(decisions)
            for level, decisions in sorted(decision_by_level.items(), key=lambda item: int(item[0]))
        },
        "target_branch_level_metrics": {
            level: branch_metrics(branches)
            for level, branches in sorted(branch_by_level.items(), key=lambda item: int(item[0]))
        },
        "sample_level_metrics": {
            level: sample_metrics(samples)
            for level, samples in sorted(sample_by_level.items(), key=lambda item: int(item[0]))
        },
    }


def argument_parser():
    parser = argparse.ArgumentParser(description="Independent depth-wise local decisions; source-record OR aggregation.")
    parser.add_argument("--tree-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ir-sglang-urls", nargs="+", default=["http://127.0.0.1:30000/v1"])
    parser.add_argument("--prompt", choices=["child-subtree", "original"], default="child-subtree")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-logprobs", type=int, default=30)
    parser.add_argument("--api-max-retries", type=int, default=1)
    parser.add_argument("--format-max-retries", type=int, default=1)
    parser.add_argument("--format-error-ranks", choices=["legacy", "none"], default="legacy",
                        help="legacy reproduces historical rank tables; none gives no rank to failed-format decisions")
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--extractor-version", choices=["legacy", "fixed"], default="fixed")
    parser.add_argument("--evaluation-mode", choices=["paper", "toy"], default="paper")
    parser.add_argument("--tree-profile", choices=["historical", "rebuilt", "toy"], default="historical")
    parser.add_argument("--expected-manifest", help="Frozen source/input-byte manifest from suite preflight")
    parser.add_argument("--run-spec", help="Resolved suite condition; validated before model requests")
    parser.add_argument("--artifact-fingerprints", default="configs/historical_tree_identity.json")
    return parser


def run_evaluation(args, *, extractor=None):
    from concurrent.futures import ThreadPoolExecutor
    from .prompts import BRANCH_PROMPT, PAPER_PROMPT
    from .protocol import (source_manifest, tree_identity, historical_tree_identity, validate_rows,
                           PARENT_COUNTS, write_json, file_sha256)
    if min(args.workers, args.max_tokens, args.top_logprobs, args.api_max_retries, args.format_max_retries) <= 0:
        raise ValueError("Numeric settings must be positive")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    expected = source_manifest(args.input_dir, mode=args.evaluation_mode)
    if args.expected_manifest:
        frozen = json.loads(Path(args.expected_manifest).read_text())
        if frozen != expected:
            raise ValueError("Evaluation inputs changed after preflight")
    identity = tree_identity(args.tree_dir)
    if (args.tree_profile == "toy") != (args.evaluation_mode == "toy"):
        raise ValueError("Toy mode requires tree-profile toy and cannot claim paper completeness")
    if args.tree_profile == "historical":
        fingerprint = json.loads(Path(args.artifact_fingerprints).read_text())
        if identity != historical_tree_identity(fingerprint):
            raise ValueError("Historical tree identity mismatch; label new trees as rebuilt")
    spec = json.loads(Path(args.run_spec).read_text()) if args.run_spec else None
    if spec:
        from .suite import validate_run_spec
        validate_run_spec(spec, args, identity, expected)
    paths = sorted(Path(args.input_dir).glob("*.json"))
    output.mkdir(parents=True)
    write_json(output / "invocation.json", vars(args))
    write_json(output / "inputs.json", expected)
    write_json(output / "protocol.json", {"run_spec": spec, "tree_identity": identity,
        "source_ids_sha256": expected["source_ids_sha256"], "tree_profile": args.tree_profile,
        "evaluation_mode": args.evaluation_mode, "model_name": args.model_name,
        "extractor_version": args.extractor_version, "format_error_ranks": args.format_error_ranks,
        "prompt_store_sha256": file_sha256(Path(__file__).with_name("prompts.json")),
        "extractor_source_sha256": file_sha256(Path(__file__).with_name("probability.py"))})
    evaluator = LevelIREvaluator(args.tree_dir, args.ir_sglang_urls,
        extractor=extractor, model_name=args.model_name, extractor_version=args.extractor_version,
        prompt_name=BRANCH_PROMPT if args.prompt == "child-subtree" else PAPER_PROMPT,
        max_tokens=args.max_tokens, top_logprobs=args.top_logprobs,
        api_max_retries=args.api_max_retries, format_max_retries=args.format_max_retries,
        format_error_ranks=args.format_error_ranks)

    def evaluate(path):
        try:
            return evaluator.evaluate_sample(str(path))
        except Exception as exc:
            return {"source_id": path.stem, "status": "error", "error": str(exc)}

    results = []
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(evaluate, paths):
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    summary = summarize(results, args.format_error_ranks)
    summary["format_error_ranks"] = args.format_error_ranks
    try:
        summary.update(validate_rows(results, expected, expected_tree=identity, actual_tree=identity,
            parent_counts=PARENT_COUNTS if args.tree_profile == "historical" else None))
    except ValueError as exc:
        summary.update(complete_five_depth_evaluation=False, paper_complete=False, validation_error=str(exc))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["complete_five_depth_evaluation"]:
        raise ValueError("Incomplete evaluation: " + summary["validation_error"])
    return summary


def main(argv=None):
    args = argument_parser().parse_args(argv)
    summary = run_evaluation(args)
    print(json.dumps(summary["sample_level_metrics"], indent=2))


if __name__ == "__main__":
    main()
