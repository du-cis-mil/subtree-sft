"""Paired source-record statistics; imports NumPy only when analysis is requested."""
import json
from pathlib import Path
import numpy as np
from .statistics import bootstrap_diff, exact_mcnemar_p
from .protocol import object_sha256


def load(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["source_id"]
            if sid in rows:
                raise ValueError(f"Duplicate source ID: {sid}")
            if row.get("status", "ok") != "ok":
                raise ValueError(f"Incomplete source record: {sid}")
            if "correct" in row:
                vals = row["correct"]
            else:
                vals = [row["sample_level"][str(i)]["is_correct_any"] for i in range(5)]
            if len(vals) != 5 or any(type(v) is not bool for v in vals):
                raise ValueError(f"Expected five boolean outcomes: {sid}")
            rows[sid] = vals
    if not rows:
        raise ValueError("Empty results")
    return rows


def compare(a, b, iterations=5000, seed=42):
    if set(a) != set(b):
        raise ValueError("Paired statistics require identical source ID sets; refusing intersection-only comparison")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    ids = sorted(a)
    rng = np.random.default_rng(seed)
    result = {}
    for i in range(5):
        av, bv = [a[s][i] for s in ids], [b[s][i] for s in ids]
        diff, low, high = bootstrap_diff(av, bv, iterations=iterations, rng=rng)
        ao, bo, p = exact_mcnemar_p(av, bv)
        result[f"D{i}"] = {"n": len(ids), "a_accuracy": sum(av)/len(ids), "b_accuracy": sum(bv)/len(ids),
            "difference_pp": 100*float(diff), "ci95_pp": [100*float(low), 100*float(high)],
            "a_only": ao, "b_only": bo, "exact_mcnemar_p": p}
    for name, depths in [("macro_D0_D3", 4), ("macro_D0_D4", 5)]:
        av = sum(v[i] for v in a.values() for i in range(depths))/(depths*len(a))
        bv = sum(v[i] for v in b.values() for i in range(depths))/(depths*len(b))
        result[name] = {"a": av, "b": bv, "difference_pp": 100*(av-bv)}
    return {"bootstrap_iterations": iterations, "bootstrap_seed": seed,
            "bootstrap_method": "paired percentile bootstrap; one generator advanced in depth order D0-D4",
            "random_generator": type(rng.bit_generator).__name__, "numpy_version": np.__version__,
            "quantile_method": "linear", "quantile_probabilities": [0.025, 0.975],
            "source_order": "lexicographically sorted source IDs",
            "source_order_sha256": object_sha256(ids), "statistics": result}


