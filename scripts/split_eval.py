#!/usr/bin/env python3
"""Convert TOMATO-Star held-out JSONL to per-source files in a new directory."""
import argparse
import json
import re
from pathlib import Path


def split(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seen = set()
    with Path(source).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["source_id"]
            if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
                raise ValueError(f"Unsafe source_id: {sid!r}")
            if sid in seen:
                raise ValueError(f"Duplicate source_id: {sid}")
            seen.add(sid)
            (output / f"{sid}.json").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    if not seen:
        raise ValueError("No evaluation records found")
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(f"Wrote {split(args.input_jsonl, args.output_dir)} evaluation records")
