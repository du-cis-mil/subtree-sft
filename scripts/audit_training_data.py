#!/usr/bin/env python3
"""Audit source-record coverage of the two ordered 10,000-example training sets."""
import argparse
from collections import Counter
import json
from pathlib import Path
from soict_ir.prompts import instruction_prompts, PAPER_PROMPT
from soict_ir.protocol import file_sha256


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def context(prompt):
    fragments = instruction_prompts(PAPER_PROMPT)
    before, sep, remaining = prompt.partition("**Research Question:**\n")
    if not sep:
        raise ValueError("Original IR prompt has no research-question marker")
    question, sep, remaining = remaining.partition(fragments[1])
    if not sep:
        raise ValueError("Original IR prompt has an unrecognized background marker")
    background, sep, _ = remaining.partition(fragments[2])
    if not sep:
        raise ValueError("Original IR prompt has an unrecognized previous-hypothesis marker")
    return question, background


def audit(source_train, paper_sft, child_sft):
    by_context = {}
    source_ids = set()
    for row in rows(source_train):
        key = row["research_question"], row["background_survey"]
        by_context.setdefault(key, set()).add(row["source_id"])
        source_ids.add(row["source_id"])
    paper_ids = set()
    paper_count = 0
    for row in rows(paper_sft):
        user = next(c["content"] for c in row["conversations"] if c["role"] == "user")
        matches = by_context.get(context(user), set())
        if len(matches) != 1:
            raise ValueError(f"Expected a unique training-source context, found {len(matches)}")
        paper_ids.update(matches)
        paper_count += 1
    child_ids = set()
    child_count = 0
    depths = Counter()
    for row in rows(child_sft):
        meta = row["metadata"]
        if meta["source_id"] not in source_ids:
            raise ValueError("Child-subtree SFT contains a source absent from the training split")
        child_ids.add(meta["source_id"])
        depths[str(meta["node_level"])] += 1
        child_count += 1
    return {"paper_selection_examples": paper_count, "child_subtree_examples": child_count,
            "paper_selection_unique_sources": len(paper_ids), "child_subtree_unique_sources": len(child_ids),
            "shared_sources": len(paper_ids & child_ids), "child_depth_distribution": dict(depths),
            "schema_version": 1, "provenance": {"kind": "provided_training_files",
                "inputs": {"source_train": file_sha256(source_train),
                           "paper_training_prefix": file_sha256(paper_sft),
                           "child_training_prefix": file_sha256(child_sft)},
                "weight_identity_verified": False}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--paper-sft", required=True)
    parser.add_argument("--child-sft", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.source_train, args.paper_sft, args.child_sft)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
