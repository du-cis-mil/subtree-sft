"""Reaggregate saved decisions without calling a model or assigning failed-format ranks."""
import json
from pathlib import Path
from statistics import mean, median
from .protocol import correctness, pseudonym


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def minimal_decisions(row, *, anonymize=False):
    """Preserve enough numeric evidence for both rank policies, with no paper text/paths."""
    values = correctness(row)
    sid = row["source_id"]
    return {"source_id": pseudonym(sid) if anonymize else sid, "correct": values,
            "decisions": [{k: d[k] for k in ("node_id", "level", "candidate_count",
                           "is_correct", "format_error", "best_correct_rank")}
                          for d in row["decisions"]]}


def source_metrics(rows, policy="legacy"):
    if policy not in {"legacy", "none"}:
        raise ValueError("Unknown rank policy")
    if not rows:
        raise ValueError("Empty results")
    ids = [r["source_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs")
    correct = [correctness(row) for row in rows]
    result = {}
    for depth in range(5):
        ranks, failures, decisions = [], 0, 0
        for row in rows:
            if "decisions" in row:
                ds = [d for d in row["decisions"] if d["level"] == depth]
                if not ds:
                    raise ValueError("Missing depth decisions")
                decisions += len(ds)
                failures += sum(d["format_error"] for d in ds)
                valid = [d["best_correct_rank"] for d in ds
                         if (policy == "legacy" or not d["format_error"]) and d["best_correct_rank"] is not None]
                rank = min(valid) if valid else None
            elif policy == "legacy":
                rank = row["best_rank"][depth] if "best_rank" in row else row["sample_level"][str(depth)]["best_correct_rank"]
            else:
                raise ValueError("Corrected ranks require decision-level format-error evidence")
            if rank is not None:
                if type(rank) is not int or rank < 1:
                    raise ValueError("Rank must be a positive integer or null")
                ranks.append(rank)
        n = len(rows)
        denominator = len(ranks) if policy == "legacy" else n
        result[str(depth)] = {"n": n, "correct_any": sum(v[depth] for v in correct),
            "accuracy_any": sum(v[depth] for v in correct) / n,
            "valid_rank_sources": len(ranks), "no_valid_rank_sources": n-len(ranks),
            "mean_valid_rank": mean(ranks) if ranks else None,
            "median_valid_rank": median(ranks) if ranks else None,
            "topk_denominator": denominator,
            "topk_any_rate": {f"top_{k}": sum(r <= k for r in ranks) / denominator if denominator else None
                              for k in [1, 2, 3, 5, 10, 15]},
            "format_errors": failures if decisions else None, "parent_decisions": decisions or None}
    return {"rank_policy": policy, "rank_summary_population": "sources with a valid rank",
            "sample_level_metrics": result}
