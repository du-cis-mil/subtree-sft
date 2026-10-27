#!/usr/bin/env python3
"""Assemble saved conditions from sequentially served models; no model calls."""
import argparse
import json
from soict_ir.reporting import merge_suites

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experiments", default="configs/experiments.json")
    a = p.parse_args()
    print(json.dumps(merge_suites(a.suite_dirs, a.output_dir, a.experiments), indent=2))
