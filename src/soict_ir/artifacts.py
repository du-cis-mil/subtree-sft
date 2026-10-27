"""Validate numeric report inputs and create portable figures using only the stdlib."""
import base64
from html import escape
import json
import math
from pathlib import Path

from .protocol import file_sha256

MAIN = ["base_child", "repro_original", "child_child"]
NAMES = ["Base", "MOOSE-Star Repro", "Child-Subtree SFT"]
COLORS = ["#777777", "#3479aa", "#e78b37"]
_NO_BUDGET_CHECK = object()


def number(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError("Invalid numeric field: " + label)


def sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Missing/invalid SHA-256: " + label)


def integer(value, label, *, positive=False):
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise ValueError("Expected " + ("positive" if positive else "nonnegative") + " integer: " + label)


def validate_inputs(tree, training, data, identity, population, *, historical=False,
                    required_training_examples=_NO_BUDGET_CHECK):
    if tree.get("tree_identity") != identity:
        raise ValueError("Granularity summary belongs to a different/unidentified tree")
    for key in ("structure", "papers", "config"):
        sha(identity.get(key), "tree_identity." + key)
    integer(tree.get("num_papers"), "tree.num_papers", positive=True)
    integer(population, "evaluation population", positive=True)
    if set(tree.get("depths", {})) != set(map(str, range(5))):
        raise ValueError("Tree summary requires every depth D0-D4")
    for depth, values in tree["depths"].items():
        for key in ("parents", "mean_children", "mean_papers_per_candidate", "median_papers_per_candidate"):
            number(values.get(key), "tree.D" + depth + "." + key, positive=True)
        if type(values["parents"]) is not int:
            raise ValueError("Tree parent counts must be integers")
    for key in ("paper_selection_examples", "child_subtree_examples", "paper_selection_unique_sources",
                "child_subtree_unique_sources", "shared_sources"):
        integer(training.get(key), "training." + key, positive=key != "shared_sources")
    # Low-level audit diagnostics may use smaller synthetic files. A paper
    # report must bind actual audited examples to the fixed SoICT recipe.
    if required_training_examples is not _NO_BUDGET_CHECK:
        if type(required_training_examples) is not int or required_training_examples != 10000:
            raise ValueError("SoICT paper recipe requires 10000 training examples per model")
        for key in ("paper_selection_examples", "child_subtree_examples"):
            if training[key] != required_training_examples:
                raise ValueError("Training audit must match the 10000-example paper recipe: " + key)
    depths = training.get("child_depth_distribution", {})
    if set(depths) != set(map(str, range(5))):
        raise ValueError("Training summary requires every depth D0-D4")
    for value in depths.values():
        if type(value) is not int or value < 0:
            raise ValueError("Training depth counts must be nonnegative integers")
    if sum(depths.values()) != training["child_subtree_examples"]:
        raise ValueError("Training depth/example counts disagree")
    for prefix in ("paper_selection", "child_subtree"):
        if training[prefix + "_unique_sources"] > training[prefix + "_examples"]:
            raise ValueError("More training sources than examples")
    if training["shared_sources"] > min(training["paper_selection_unique_sources"], training["child_subtree_unique_sources"]):
        raise ValueError("Invalid shared training source count")
    provenance = training.get("provenance", {})
    if provenance.get("kind") not in {"historical_training_prefixes", "provided_training_files", "synthetic_fixture"}:
        raise ValueError("Training summary needs explicit input provenance")
    for key in ("source_train", "paper_training_prefix", "child_training_prefix"):
        sha(provenance.get("inputs", {}).get(key), "training." + key)
    if data is None:
        return
    if data.get("schema_version") != 1 or data.get("audit_kind") != "raw_corpus_and_leaf_paths":
        raise ValueError("Invalid corpus audit schema")
    if data.get("tree_identity") != identity:
        raise ValueError("Corpus audit tree identity mismatch")
    if data.get("granularity") != tree:
        raise ValueError("Corpus audit and granularity summary disagree")
    config = data.get("tree_config", {})
    for key in ("embedding_dim", "branching_factor", "max_depth"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError("Missing/invalid audited tree configuration: " + key)
    if config.get("use_medoid") != 1:
        raise ValueError("Paper report requires medoid representatives")
    required = ("train_rows", "test_rows", "test_inspiration_seen", "within_source_deduplicated_titles",
        "within_source_duplicate_annotations", "distinct_test_normalized_titles", "num_unique_papers",
        "training_derived_corpus_papers", "test_inspiration_added", "tree_paper_records", "tree_leaf_count",
        "matched_targets_within_source", "sources_with_mapped_target", "distinct_matched_leaf_papers",
        "unmatched_annotations", "duplicate_matched_annotations")
    counts = data.get("counts", {})
    if not isinstance(counts, dict):
        raise ValueError("Invalid corpus counts object")
    for key in set(required) | set(counts):
        integer(counts.get(key), "corpus." + key)
    for key in ("source_train_sha256", "source_test_sha256"):
        sha(data.get("inputs", {}).get(key), "corpus." + key)
    if data["inputs"]["source_train_sha256"] != provenance["inputs"]["source_train"]:
        raise ValueError("Training/corpus source hashes disagree")
    if counts["test_rows"] != population or counts["tree_leaf_count"] != tree["num_papers"]:
        raise ValueError("Corpus/tree/population counts disagree")
    if counts["num_unique_papers"] != tree["num_papers"] or counts["tree_paper_records"] != tree["num_papers"]:
        raise ValueError("Reconstructed corpus differs from tree population")
    if counts["within_source_deduplicated_titles"] + counts["within_source_duplicate_annotations"] != counts["test_inspiration_seen"]:
        raise ValueError("Corpus duplicate accounting disagrees")
    if counts["matched_targets_within_source"] + counts["duplicate_matched_annotations"] + counts["unmatched_annotations"] != counts["test_inspiration_seen"]:
        raise ValueError("Corpus mapping accounting disagrees")
    if counts["training_derived_corpus_papers"] + counts["test_inspiration_added"] != counts["num_unique_papers"]:
        raise ValueError("Corpus increment accounting disagrees")
    # Deduplication within a source and across sources count different things.
    # Shared targets across queries are valid; neither count may exceed its
    # defining population. Invalid audits are never manuscript corrections.
    bounds = (
        ("sources_with_mapped_target", "test_rows"),
        ("sources_with_mapped_target", "matched_targets_within_source"),
        ("matched_targets_within_source", "within_source_deduplicated_titles"),
        ("distinct_matched_leaf_papers", "matched_targets_within_source"),
        ("distinct_matched_leaf_papers", "distinct_test_normalized_titles"),
        ("distinct_matched_leaf_papers", "tree_leaf_count"),
        ("distinct_test_normalized_titles", "within_source_deduplicated_titles"),
        ("test_inspiration_added", "distinct_test_normalized_titles"),
    )
    for smaller, larger in bounds:
        if counts[smaller] > counts[larger]:
            raise ValueError("Invalid corpus count relationship: " + smaller + " > " + larger)
    if counts["within_source_deduplicated_titles"] > counts["test_rows"] * counts["distinct_test_normalized_titles"]:
        raise ValueError("Corpus within-source title counts exceed their source/title populations")
    if counts["matched_targets_within_source"] > counts["sources_with_mapped_target"] * counts["distinct_matched_leaf_papers"]:
        raise ValueError("Corpus within-source targets exceed their mapped source/leaf populations")
    # Counter-based older audits omit zero-valued training increment fields.
    # Check the additional accounting whenever those fields are supplied.
    if "train_source_added" in counts or "train_inspiration_added" in counts:
        if counts.get("train_source_added", 0) > counts["train_rows"] or \
           counts.get("train_inspiration_added", 0) > counts.get("train_inspiration_seen", 0):
            raise ValueError("Corpus training additions exceed observed records/annotations")
        if counts.get("train_source_added", 0) + counts.get("train_inspiration_added", 0) != counts["training_derived_corpus_papers"]:
            raise ValueError("Corpus training increment accounting disagrees")
    if bool(counts["matched_targets_within_source"]) != bool(counts["sources_with_mapped_target"]) or \
       bool(counts["matched_targets_within_source"]) != bool(counts["distinct_matched_leaf_papers"]):
        raise ValueError("Corpus mapped target/source/leaf counts disagree on emptiness")
    for prefix in ("paper_selection", "child_subtree"):
        if training[prefix + "_unique_sources"] > counts["train_rows"]:
            raise ValueError("Training source coverage exceeds corpus training records")
    if training["paper_selection_unique_sources"] + training["child_subtree_unique_sources"] - training["shared_sources"] > counts["train_rows"]:
        raise ValueError("Training source union exceeds corpus training records")
    parents = data.get("parent_counts", {})
    if set(parents) != set(map(str, range(5))) or any(type(n) is not int or
            not 0 <= n <= counts["matched_targets_within_source"] for n in parents.values()):
        raise ValueError("Invalid corpus parent counts")
    if parents["0"] != counts["sources_with_mapped_target"]:
        raise ValueError("Corpus root decisions and mapped source counts disagree")
    if historical and [parents[str(d)] for d in range(5)] != [1658, 2575, 2895, 3027, 3081]:
        raise ValueError("Historical corpus parent counts disagree")


def population_gaps(data, population):
    """Consistent but incomplete mappings can be shown only in partial reports.

    Evaluation protocols separately validate every source at all five depths;
    aggregate parent counts alone cannot establish per-source depth coverage.
    """
    gaps = {}
    if data["counts"]["sources_with_mapped_target"] != population:
        gaps["target_mapping"] = "Not every evaluation source has a mapped target"
    for depth, count in data["parent_counts"].items():
        if count < population:
            gaps["depth_coverage:D" + depth] = "Insufficient parent decisions for all sources at D" + depth
    return gaps


def validate_figure_assets(project_root):
    root = Path(project_root)
    manifest = json.loads((root / "docs/figures/manifest.json").read_text())
    allowed = {"docs/figures/figure_child_subtree_selection_supervision_innerlabels_transparent.png",
               "docs/figures/figure_candidate_granularity_compact_transparent.png"}
    if {a["path"] for a in manifest["assets"]} != allowed or len(manifest["assets"]) != 2:
        raise ValueError("Expected exactly the manuscript's two concept panels")
    for asset in manifest["assets"]:
        path = root / asset["path"]
        if path.is_symlink() or path.stat().st_size != asset["bytes"] or not 0 < asset["bytes"] < 5_000_000:
            raise ValueError("Invalid figure asset size/symlink")
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") or file_sha256(path) != asset["sha256"]:
            raise ValueError("Figure PNG signature/hash mismatch")
    return manifest


def make_figures(directory, accuracy, assets, project_root):
    """The six x categories and three series match the paper, including Overall."""
    directory = Path(directory)
    figure = [{"depth": category, **{n: accuracy[n][category] for n in MAIN}}
              for category in ["D0", "D1", "D2", "D3", "D4", "Overall"]]
    for row in figure:
        for name in MAIN:
            value = row[name]
            if value is not None:
                number(value, "accuracy." + name + "." + row["depth"])
                if value > 100:
                    raise ValueError("Accuracy must be in [0, 100]")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="550" viewBox="0 0 1050 550">',
           '<title>Depth-wise Top-1 accuracy and Overall, three manuscript models</title>',
           '<rect width="1050" height="550" fill="white"/>']
    # Leave space above 100% for value labels and the separate legend.
    for tick in (0, 20, 40, 60, 80, 100):
        y = 465 - tick*3.6
        svg += [f'<path d="M70 {y} H1020" stroke="#dddddd"/>',
                f'<text x="58" y="{y+5}" text-anchor="end">{tick}</text>']
    svg += ['<text x="20" y="285" transform="rotate(-90 20 285)" text-anchor="middle">Top-1 accuracy (%)</text>']
    for index, row in enumerate(figure):
        x = 95 + index*155
        svg.append(f'<text x="{x+45}" y="490" text-anchor="middle">{row["depth"]}</text>')
        for series, name in enumerate(MAIN):
            value = row[name]
            if value is None:
                svg.append(f'<text x="{x+series*30}" y="455">NA</text>')
                continue
            y = 465 - value*3.6
            svg += [f'<rect data-series="{name}" data-category="{row["depth"]}" data-value="{value}" x="{x+series*30}" y="{y}" width="24" height="{value*3.6}" fill="{COLORS[series]}"/>',
                    f'<text data-value-label="{name}" data-category="{row["depth"]}" x="{x+series*30+12}" y="{y-7}" text-anchor="middle" font-size="12">{value:.2f}</text>']
    for i, name in enumerate(NAMES):
        svg += [f'<rect x="{100+i*305}" y="25" width="18" height="18" fill="{COLORS[i]}"/>',
                f'<text x="{125+i*305}" y="40">{escape(name)}</text>']
    svg.append('</svg>')
    (directory / "depthwise_figure.svg").write_text("\n".join(svg)+"\n")
    tex = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}",
           r"\begin{document}\begin{tikzpicture}",
           r"\begin{axis}[ybar,bar width=9pt,ymin=0,ymax=110,ytick={0,20,40,60,80,100},ylabel={Top-1 accuracy (\%)},symbolic x coords={D0,D1,D2,D3,D4,Overall},xtick=data,unbounded coords=jump,legend style={at={(0.5,1.05)},anchor=south,legend columns=3}]"]
    for name, label in zip(MAIN, NAMES):
        coords = " ".join("(" + row["depth"] + "," + ("nan" if row[name] is None else str(row[name])) + ")" for row in figure)
        tex += [r"\addplot coordinates {" + coords + "};", r"\addlegendentry{" + label + "}"]
        for row in figure:
            if row[name] is None:
                tex.append(r"\node[font=\tiny,rotate=90,anchor=west] at (axis cs:" + row['depth'] + ",3) {" + label + ": NA};")
    tex += [r"\end{axis}\end{tikzpicture}\end{document}"]
    (directory / "depthwise_figure.tex").write_text("\n".join(tex)+"\n")
    # Self-contained concept-figure view: no links to the parent repository.
    panels = []
    for a in assets["assets"]:
        encoded = base64.b64encode((Path(project_root) / a["path"]).read_bytes()).decode()
        panels.append('<figure><img alt="Panel '+a["panel"]+'" src="data:image/png;base64,'+encoded+'"/><figcaption>('+a["panel"]+') '+
                      ("Supervision for child-subtree selection" if a["panel"] == "a" else "Candidate granularity across the tree")+'</figcaption></figure>')
    (directory / "method_overview.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Method overview</title><style>body{font-family:sans-serif}main{display:flex}figure{width:48%;margin:1%}img{width:100%}</style><main>'+''.join(panels)+'</main><p>Child subtrees are represented by medoid papers. Conceptual illustration; no experimental measurements.</p><p>Figure redistribution terms require author confirmation; see docs/figures/manifest.json.</p></html>\n')
    # Copy the exact assets into the output for a relocatable standalone TeX file.
    import shutil
    for i, a in enumerate(assets["assets"]):
        shutil.copyfile(Path(project_root) / a["path"], directory / ("method_panel_" + str(i+1) + ".png"))
    (directory / "method_overview.tex").write_text(r"""\documentclass{article}
\usepackage{graphicx}
\begin{document}
\begin{figure}\centering
\begin{minipage}{.48\linewidth}\includegraphics[width=\linewidth]{method_panel_1.png}\\(a) Supervision for child-subtree selection\end{minipage}\hfill
\begin{minipage}{.48\linewidth}\includegraphics[width=\linewidth]{method_panel_2.png}\\(b) Candidate granularity across the tree\end{minipage}
\caption{Overview of the proposed supervision for child-subtree selection and candidate granularity across a hierarchical tree. Child subtrees are represented by medoid papers in our experiments.}
\end{figure}\end{document}
""")
    return figure
