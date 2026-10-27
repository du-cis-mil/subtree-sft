#!/usr/bin/env python3
"""Compute candidate-subtree granularity over all internal nodes of a saved tree."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
from soict_ir.tree import HierarchicalSearchTree
from soict_ir.protocol import tree_identity


def granularity(tree):
    by_level = defaultdict(lambda: {"parents": 0, "children": [], "paper_counts": []})

    def visit(node):
        if node.get("is_leaf"):
            return 1
        counts = [visit(child) for child in node.get("children", [])]
        if len(counts) > 1:
            level = by_level[str(node.get("level", 0))]
            level["parents"] += 1
            level["children"].append(len(counts))
            level["paper_counts"].extend(counts)
        return sum(counts)

    total = visit(tree.root)
    return {"num_papers": total, "depths": {
        depth: {"parents": values["parents"], "mean_children": mean(values["children"]),
                "mean_papers_per_candidate": mean(values["paper_counts"]),
                "median_papers_per_candidate": median(values["paper_counts"])}
        for depth, values in sorted(by_level.items(), key=lambda kv: int(kv[0]))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = granularity(HierarchicalSearchTree.load(args.tree_dir))
    result["tree_identity"] = tree_identity(args.tree_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
