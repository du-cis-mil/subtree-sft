"""CPU-only identities and completeness checks shared by runners and reports."""
import hashlib
import json
from pathlib import Path

PAPER_N = 1658
PARENT_COUNTS = [1658, 2575, 2895, 3027, 3081]
CONDITIONS = {
    "base_child": ("base", "child-subtree"),
    "public_child": ("public", "child-subtree"),
    "public_original": ("public", "original"),
    "repro_child": ("repro", "child-subtree"),
    "repro_original": ("repro", "original"),
    "child_child": ("child", "child-subtree"),
    "child_original": ("child", "original"),
}
TREE_FILES = {"structure": "hierarchical_tree.json", "papers": "papers.json", "config": "tree_config.json"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pseudonym(source_id):
    return hashlib.sha256(source_id.encode()).hexdigest()[:20]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tree_identity(tree_dir):
    return {key: file_sha256(Path(tree_dir) / name) for key, name in TREE_FILES.items()}


def historical_tree_identity(fingerprints):
    return {key: fingerprints["evaluation_tree_" + key]["sha256"]
            for key in TREE_FILES}


def source_manifest(input_dir, *, mode="paper"):
    """Freeze source IDs AND input bytes before any model request."""
    paths = sorted(Path(input_dir).glob("*.json"))
    ids, files = [], {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        sid = row.get("source_id") or path.stem
        if not isinstance(sid, str) or not sid or sid != path.stem:
            raise ValueError(f"source_id must equal input filename stem: {path.name}")
        ids.append(sid)
        files[path.name] = file_sha256(path)
    return expected_sources(ids, mode=mode, input_files=files)


def expected_sources(ids, *, mode="paper", input_files=None):
    if mode not in {"paper", "toy"}:
        raise ValueError("mode must be paper or toy")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected nonempty, unique source IDs")
    if mode == "paper" and len(ids) != PAPER_N:
        raise ValueError(f"Paper evaluation requires {PAPER_N} sources, got {len(ids)}")
    result = {"mode": mode, "count": len(ids), "source_ids": sorted(ids),
              "source_ids_sha256": object_sha256(sorted(ids))}
    if input_files is not None:
        result["input_files_sha256"] = object_sha256(input_files)
    return result


def correctness(row):
    if row.get("status", "ok") != "ok":
        raise ValueError(f"Incomplete/API-failed source: {row.get('source_id')}")
    if "correct" in row:
        values = row["correct"]
    else:
        values = [row.get("sample_level", {}).get(str(i), {}).get("is_correct_any") for i in range(5)]
    if len(values) != 5 or any(type(v) is not bool for v in values):
        raise ValueError(f"Expected five boolean depth outcomes: {row.get('source_id')}")
    return values


def validate_rows(rows, expected, *, expected_tree=None, actual_tree=None, parent_counts=None):
    """A partial set cannot pass merely because all supplied rows are complete."""
    rebuilt = expected_sources(expected["source_ids"], mode=expected["mode"])
    if expected["source_ids_sha256"] != rebuilt["source_ids_sha256"] or expected["count"] != rebuilt["count"]:
        raise ValueError("Corrupt expected source manifest")
    ids = [r["source_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate result source IDs")
    if set(ids) != set(expected["source_ids"]):
        raise ValueError("Result source set differs from the frozen expected source set")
    if expected_tree is not None and actual_tree != expected_tree:
        raise ValueError("Evaluation tree identity mismatch")
    totals = [0] * 5
    for row in rows:
        vals = correctness(row)
        if "decisions" not in row:
            if parent_counts is not None:
                raise ValueError("Parent count validation requires decision-level evidence")
            continue
        seen = set()
        for decision in row["decisions"]:
            level = decision["level"]
            key = (level, decision["node_id"])
            if type(level) is not int or level not in range(5) or key in seen:
                raise ValueError("Invalid/duplicate ground-truth parent decision")
            seen.add(key)
            totals[level] += 1
            if type(decision.get("is_correct")) is not bool or type(decision.get("format_error")) is not bool:
                raise ValueError("Decision correctness/format flags must be boolean")
        for depth in range(5):
            ds = [d for d in row["decisions"] if d["level"] == depth]
            if not ds or any(d["is_correct"] for d in ds) != vals[depth]:
                raise ValueError("Missing depth or inconsistent source OR aggregation")
    if parent_counts is not None and totals != parent_counts:
        raise ValueError(f"Parent counts mismatch: {totals} != {parent_counts}")
    return {"records": len(rows), "source_ids_sha256": rebuilt["source_ids_sha256"],
            "parent_counts": totals, "complete_five_depth_evaluation": True,
            "paper_complete": expected["mode"] == "paper"}
