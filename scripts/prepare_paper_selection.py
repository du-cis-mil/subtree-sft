#!/usr/bin/env python3
"""Copy the ordered prefix of the ORIGINAL released IR SFT data, without reformatting."""
import argparse
import hashlib
import json
from pathlib import Path
from soict_ir.supervision import write_dataset_info


def prepare(source, output, count=10000):
    if count <= 0:
        raise ValueError("count must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    written = 0
    digest = hashlib.sha256()
    with Path(source).open("rb") as src, (output / "train.jsonl").open("wb") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if "conversations" not in row:
                raise ValueError("Expected original IR SFT conversations, not raw source records")
            dst.write(line)
            digest.update(line)
            written += 1
            if written == count:
                break
    write_dataset_info(str(output), "train", "train.jsonl")
    (output / "summary.json").write_text(json.dumps({"selection": "original ordered prefix",
        "num_examples": written, "sha256": digest.hexdigest()}, indent=2) + "\n")
    if written != count:
        raise ValueError(f"Requested {count} examples, found {written}; output is incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=10000)
    args = parser.parse_args()
    prepare(args.input_jsonl, args.output_dir, args.max_examples)
