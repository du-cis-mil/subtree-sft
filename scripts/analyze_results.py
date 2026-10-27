#!/usr/bin/env python3
"""Paired source-record bootstrap and exact McNemar tests; reject incomplete pairs."""
import argparse
import json
from pathlib import Path
from soict_ir.analysis import load, compare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Model A results.jsonl or compact outcomes.jsonl")
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare(load(args.a), load(args.b), args.bootstrap_iterations, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
