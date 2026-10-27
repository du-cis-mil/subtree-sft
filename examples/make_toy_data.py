#!/usr/bin/env python3
"""Write entirely synthetic papers and queries for CPU protocol tests."""
import argparse
import json
from pathlib import Path


def make(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tree_dir = output / "toy_tree"
    tree_dir.mkdir()
    papers = [{"paper_id": f"p{i}", "title": f"Synthetic paper {i}",
               "abstract": f"Invented abstract for toy paper {i}."} for i in range(1, 5)]

    def leaf(i):
        return {"node_id": f"leaf{i}", "level": 2, "is_leaf": True, "paper_id": f"p{i}",
                "centroid_paper_id": f"p{i}", "num_papers": 1, "children": []}

    left = {"node_id": "left", "level": 1, "is_leaf": False, "centroid_paper_id": "p1",
            "num_papers": 2, "children": [leaf(1), leaf(2)]}
    right = {"node_id": "right", "level": 1, "is_leaf": False, "centroid_paper_id": "p3",
             "num_papers": 2, "children": [leaf(3), leaf(4)]}
    root = {"node_id": "root", "level": 0, "is_leaf": False, "centroid_paper_id": "p1",
            "num_papers": 4, "children": [left, right]}
    for name, value in [("hierarchical_tree.json", root), ("papers.json", papers),
                        ("tree_config.json", {"seed": 42, "branching_factor": 2, "num_papers": 4})]:
        (tree_dir / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    records = [{"source_id": "toy_1", "research_question": "A synthetic research problem",
        "background_survey": "Synthetic background.", "inspiration": [
            {"found_title": " SYNTHETIC   paper 3 "}, {"found_title": "Synthetic paper 1"},
            {"found_title": "Synthetic paper 3"}, {"found_title": "Missing synthetic paper"}]}]
    (output / "train.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    eval_dir = output / "eval"
    eval_dir.mkdir()
    (eval_dir / "toy_1.json").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    return tree_dir, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    print(make(parser.parse_args().output_dir)[0])
