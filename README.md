# Child-Subtree SFT

Code for **From Paper Selection to Child-Subtree Selection: Task-Aligned SFT for Hierarchical Scientific Literature Retrieval** (SoICT 2026).

[Experiment protocol](docs/reproduction.md) · [Data preparation](docs/data.md) · [Figures and tables](docs/manuscript_coverage.md)

This repository implements task-aligned supervision for hierarchical scientific literature retrieval. It constructs Paper-Selection and Child-Subtree SFT datasets, trains two models with matched budgets, and evaluates their local decisions at each depth of a hierarchical search tree.

## Setup

Use Python 3.11 for data preparation and analysis:

```bash
git clone https://github.com/du-cis-mil/subtree-sft.git
cd subtree-sft
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
```

The [experiment protocol](docs/reproduction.md) describes the separate environments for training and serving, dataset preparation, tree construction, and evaluation.

## Experiments

| Condition | Model | Evaluation prompt |
|---|---|---|
| `base_child` | Backbone | Child-Subtree |
| `public_child` | Public MOOSE-Star IR | Child-Subtree |
| `public_original` | Public MOOSE-Star IR | Original IR |
| `repro_child` | Paper-Selection SFT | Child-Subtree |
| `repro_original` | Paper-Selection SFT | Original IR |
| `child_child` | Child-Subtree SFT | Child-Subtree |
| `child_original` | Child-Subtree SFT | Original IR |

Each trained model uses the same final checkpoint for its two evaluation formats. Both SFT datasets contain 10,000 examples, and evaluation uses 1,658 held-out source records.

Prepare the data and trees, train the models, and set checkpoint paths and server endpoints in a copy of `configs/models.example.json`. Preview all seven conditions:

```bash
python scripts/run_paper_experiments.py \
  --condition all \
  --model-map configs/models.example.json \
  --plan-output outputs/plan.json
```

Start the corresponding model server, then add `--execute` to run an evaluation. Models can be served sequentially; the protocol includes commands for merging their evaluation suites.

## Generate figures, tables, and statistics

```bash
python scripts/make_paper_report.py \
  --suite-dir outputs/paper_merged \
  --output-dir outputs/report \
  --tree-summary outputs/tree_granularity.json \
  --training-audit outputs/training_source_audit.json \
  --corpus-audit outputs/corpus_audit.json
```

The report includes two figures, four tables, depth-wise accuracy, source-level Top-k metrics, exact McNemar tests, and paired bootstrap confidence intervals. The [output inventory](docs/manuscript_coverage.md) maps these artifacts and 159 report fields to the paper.

Report validation checks all seven conditions, source and depth coverage, model and tree settings, training budgets, and corpus counts. Use `--allow-partial` to inspect an incomplete experiment; missing values are marked as `NA`.

## Tests

```bash
python -m unittest discover -s tests -v
python scripts/check_synthetic_report.py --output-dir outputs/synthetic_report
python scripts/make_release.py --check
```

The CPU tests use synthetic data and mock model responses. The report example exercises the reporting pipeline with 10 bootstrap draws.

## License and citation

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for code licensing and attribution. Citation metadata is available in [CITATION.cff](CITATION.cff).
