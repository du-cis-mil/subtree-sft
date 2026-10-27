"""Finite CPU audit of corpus construction and ground-truth paths; no model imports."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

from .corpus import iter_jsonl, normalize_title
from .protocol import file_sha256, tree_identity, write_json, object_sha256, pseudonym


def audit(train_jsonl, test_jsonl, tree_dir):
    seen, counts = set(), Counter()

    def add(title, abstract, key):
        title, abstract = (title or "").strip(), (abstract or "").strip()
        if not title or not abstract:
            return
        normalized = normalize_title(title)
        if normalized not in seen:
            seen.add(normalized)
            counts[key] += 1

    for row in iter_jsonl(Path(train_jsonl)):
        counts["train_rows"] += 1
        add(row.get("title"), row.get("abstract"), "train_source_added")
        for inspiration in row.get("inspiration", []):
            counts["train_inspiration_seen"] += 1
            add(inspiration.get("found_title") or inspiration.get("supposed_title"),
                inspiration.get("found_abstract"), "train_inspiration_added")
    training_corpus_size = len(seen)
    test = list(iter_jsonl(Path(test_jsonl)))
    source_ids = [r["source_id"] for r in test]
    if not test or len(set(source_ids)) != len(test):
        raise ValueError("Test sources must be nonempty and unique")
    counts["test_rows"] = len(test)
    all_titles = set()
    for row in test:
        titles = []
        for inspiration in row.get("inspiration", []):
            counts["test_inspiration_seen"] += 1
            title = inspiration.get("found_title") or inspiration.get("supposed_title") or ""
            titles.append(normalize_title(title))
            add(title, inspiration.get("found_abstract"), "test_inspiration_added")
        all_titles.update(titles)
        counts["within_source_deduplicated_titles"] += len(set(titles))
        counts["within_source_duplicate_annotations"] += len(titles) - len(set(titles))
    counts["distinct_test_normalized_titles"] = len(all_titles)
    counts["num_unique_papers"] = len(seen)
    counts["training_derived_corpus_papers"] = training_corpus_size
    del seen

    directory = Path(tree_dir)
    papers = json.loads((directory / "papers.json").read_text())
    # Match the evaluator's normalized-title index, including its last-entry rule.
    title_index = {normalize_title(p.get("title", "")): p["paper_id"] for p in papers}
    counts["tree_paper_records"] = len(papers)
    del papers
    target_paths, requested = {}, {title_index[t] for t in all_titles if t in title_index}
    root = json.loads((directory / "hierarchical_tree.json").read_text())
    levels = defaultdict(lambda: {"parents": 0, "children": [], "papers": []})

    def walk(node, ancestors):
        if node.get("is_leaf"):
            pid = node["paper_id"]
            if pid in requested:
                if pid in target_paths:
                    raise ValueError("Target paper occurs in multiple leaves")
                target_paths[pid] = ancestors
            return 1
        children = node.get("children", [])
        decision = (node["level"], node["node_id"]) if len(children) > 1 else None
        path = ancestors + ([decision] if decision else [])
        sizes = [walk(child, path) for child in children]
        if decision:
            level = levels[str(node["level"])]
            level["parents"] += 1
            level["children"].append(len(children))
            level["papers"].extend(sizes)
        return sum(sizes)

    counts["tree_leaf_count"] = walk(root, [])
    del root
    parents = Counter()
    matched_distinct = set()
    for row in test:
        matched = set()
        for inspiration in row.get("inspiration", []):
            title = normalize_title(inspiration.get("found_title") or inspiration.get("supposed_title") or "")
            pid = title_index.get(title)
            if pid not in target_paths:
                counts["unmatched_annotations"] += 1
            elif pid in matched:
                counts["duplicate_matched_annotations"] += 1
            else:
                matched.add(pid)
        counts["matched_targets_within_source"] += len(matched)
        counts["sources_with_mapped_target"] += bool(matched)
        matched_distinct.update(matched)
        nodes = set(pair for pid in matched for pair in target_paths[pid])
        parents.update(str(depth) for depth, _ in nodes)
    counts["distinct_matched_leaf_papers"] = len(matched_distinct)
    for key in ("unmatched_annotations", "duplicate_matched_annotations", "test_inspiration_added"):
        counts.setdefault(key, 0)
    identity = tree_identity(directory)
    config = json.loads((directory / "tree_config.json").read_text())
    return {
        "schema_version": 1, "audit_kind": "raw_corpus_and_leaf_paths",
        "counts": dict(counts), "parent_counts": {str(d): parents[str(d)] for d in range(5)},
        "tree_identity": identity, "tree_config": config,
        "inputs": {"source_train_sha256": file_sha256(train_jsonl),
                   "source_test_sha256": file_sha256(test_jsonl)},
        "source_sets": {"original_ids_sha256": object_sha256(sorted(source_ids)),
                        "pseudonymous_ids_sha256": object_sha256(sorted(pseudonym(s) for s in source_ids))},
        "temporal_split": {"status": "method_description",
            "declared_train": "January 2020 through September 2025", "declared_test": "October 2025",
            "verification": "Publication-month boundaries are not inferred from source identifiers."},
        "definitions": {"test_inspiration_added": "New valid-title/abstract papers after all training sources and inspirations, using corpus builder order",
            "within_source_deduplicated_titles": "Normalized titles deduplicated separately within each test record",
            "matched_targets_within_source": "Distinct mapped leaf paper IDs summed over test sources",
            "distinct_matched_leaf_papers": "Distinct mapped leaf paper IDs across all held-out annotations"},
        "granularity": {"num_papers": counts["tree_leaf_count"], "tree_identity": identity,
            "depths": {d: {"parents": v["parents"], "mean_children": mean(v["children"]),
                "mean_papers_per_candidate": mean(v["papers"]), "median_papers_per_candidate": median(v["papers"])}
                for d, v in sorted(levels.items())}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-test", required=True)
    parser.add_argument("--tree-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tree-summary")
    args = parser.parse_args()
    for path in (args.output, args.tree_summary):
        if path and Path(path).exists():
            raise FileExistsError(path)
    result = audit(args.source_train, args.source_test, args.tree_dir)
    if args.tree_summary:
        write_json(args.tree_summary, result["granularity"])
    write_json(args.output, result)
    print(json.dumps({"counts": result["counts"], "parent_counts": result["parent_counts"]}, indent=2))


if __name__ == "__main__":
    main()
