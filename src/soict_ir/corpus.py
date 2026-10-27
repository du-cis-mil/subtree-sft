# Adapted for the standalone SoICT release (2026): local imports and CLI paths revised.
# MOOSE-Star-derived code; see LICENSE and NOTICE.
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


def normalize_title(title: str) -> str:
    return " ".join((title or "").lower().strip().split())


def paper_id(title: str, abstract: str) -> str:
    return hashlib.md5(f"{title}|||{abstract}".encode()).hexdigest()[:12]


def source_year(source_id: str) -> Optional[int]:
    try:
        value = (source_id or "").split("_")[0]
        return int(value) if value and value != "0000" else 2020
    except (ValueError, IndexError):
        return None


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def add_paper(
    papers: Dict[str, Dict],
    title: str,
    abstract: str,
    year: Optional[int],
    source_split: str,
    source_kind: str,
) -> bool:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if not title or not abstract:
        return False

    key = normalize_title(title)
    if key in papers:
        papers[key]["sources"].append({"split": source_split, "kind": source_kind})
        return False

    papers[key] = {
        "paper_id": paper_id(title, abstract),
        "title": title,
        "abstract": abstract,
        "year": year,
        "sources": [{"split": source_split, "kind": source_kind}],
    }
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    papers: Dict[str, Dict] = {}
    counts = {
        "train_rows": 0,
        "test_rows": 0,
        "train_source_added": 0,
        "train_inspiration_added": 0,
        "test_inspiration_added": 0,
        "train_inspiration_seen": 0,
        "test_inspiration_seen": 0,
    }

    for row in iter_jsonl(Path(args.train_jsonl)):
        counts["train_rows"] += 1
        year = source_year(row.get("source_id", ""))
        if add_paper(papers, row.get("title", ""), row.get("abstract", ""), year, "train", "source"):
            counts["train_source_added"] += 1
        for insp in row.get("inspiration", []):
            counts["train_inspiration_seen"] += 1
            if add_paper(
                papers,
                insp.get("found_title", "") or insp.get("supposed_title", ""),
                insp.get("found_abstract", ""),
                year,
                "train",
                "inspiration",
            ):
                counts["train_inspiration_added"] += 1

    for row in iter_jsonl(Path(args.test_jsonl)):
        counts["test_rows"] += 1
        year = source_year(row.get("source_id", ""))
        for insp in row.get("inspiration", []):
            counts["test_inspiration_seen"] += 1
            if add_paper(
                papers,
                insp.get("found_title", "") or insp.get("supposed_title", ""),
                insp.get("found_abstract", ""),
                year,
                "test",
                "inspiration",
            ):
                counts["test_inspiration_added"] += 1

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for paper in papers.values():
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    counts["num_unique_papers"] = len(papers)
    Path(args.summary).write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(papers)} unique papers to {output}")


if __name__ == "__main__":
    main()
