# SoICT hierarchical inspiration retrieval

Standalone code for all experiments in *From Paper Selection to Child-Subtree Selection: Task-Aligned SFT for Hierarchical Scientific Literature Retrieval*.

[日本語](README.ja.md) · [Protocol](docs/reproduction.md) · [Data](docs/data.md) · [Output inventory](docs/manuscript_coverage.md)

This is a code-only repository prepared for review. Historical experimental results, scores, output tables/charts, source IDs and model weights are not distributed. The code prepares two budget-matched SFT datasets, evaluates all seven conditions, and generates all 159 output fields, two figures, four tables and paired statistics from your own inputs. Later routing/teacher research is outside this artifact.

## CPU checks

From this project's root, without installation:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_paper_experiments.py --condition all --plan-output outputs/plan.json
python scripts/make_release.py --check
```

Tests use synthetic data and block real model connections. Analysis tests require NumPy; they are explicitly skipped when it is unavailable. For a new analysis environment, use Python >=3.10 and install the analysis extra (`python -m pip install -e '.[analysis]'`). Training and serving use separate new environments; follow the protocol.

## All paper experiments

| Condition | Model | Prompt |
|---|---|---|
| base_child | Backbone | Child-Subtree |
| public_child | Public MOOSE-Star IR | Child-Subtree |
| public_original | Same public checkpoint | Original IR |
| repro_child | Paper-Selection SFT final checkpoint | Child-Subtree |
| repro_original | Same Paper-Selection checkpoint | Original IR |
| child_child | Child-Subtree SFT final checkpoint | Child-Subtree |
| child_original | Same Child-Subtree checkpoint | Original IR |

The runner defaults to dry-run. Explicit `--execute` and manually prepared servers are required for real evaluation. See [the full workflow](docs/reproduction.md), including sequential serving and suite merging.

## Report your own results

```bash
PYTHONPATH=src python scripts/make_paper_report.py \
  --suite-dir outputs/paper_merged --output-dir outputs/report \
  --tree-summary outputs/tree_granularity.json \
  --training-audit outputs/training_audit.json --corpus-audit outputs/corpus_audit.json
```

The full report validates all seven evaluations, the source set, every depth, tree/model settings, corpus accounting, both 10,000-example training audits, all 159 fixed fields and every required output. `--allow-partial` explicitly reports missing evidence; `--skip-statistics` also makes the report partial. No historical reference values are supplied or substituted. `historical_numeric_match` remains false (not compared). A complete synthetic report proves code paths only.

Public Top-k excludes failed-format parent ranks but retains the source denominator. Mean/median ranks are conditional on valid ranks. Top-1 uses actual source-level OR correctness. Evaluation is independent at each ground-truth-path depth, not end-to-end traversal. Bootstrap uncertainty concerns source sampling, not training seeds.

Exact historical tree/checkpoint download URLs remain unavailable. Rebuilding is a new realization, not proof of historical byte/weight identity. Repro Child historical values and the old manuscript CI procedure remain unverified. A scaled run used 100 examples for each SFT and 17 sources per condition with SDK 1 and SGLang on a single GPU. Full-scale reproduction, a fresh environment and multi-GPU execution remain unverified; the scaled run used an API retry limit of 10 instead of the planned 1. The concept panels are review assets with author-specific redistribution terms pending; Apache-2.0 does not automatically cover them.

## Prepare an isolated code archive

```bash
python scripts/make_release.py --output /tmp/soict-code-v0.5.1-code-only.zip
```

Extract outside the parent research repository and review the extracted directory before creating a new repository. This code folder contains source, configuration, documentation and synthetic tests only. Historical results and old archives are stored separately. Keep the parent research Git history out of the new repository. Historical results and internal evidence remain outside this repository.

Synthetic report / 合成データのreport層確認（実験結果ではありません）:

```bash
PYTHONPATH=src python scripts/check_synthetic_report.py --output-dir outputs/synthetic_report
```

This uses a report-layer mock and 10 bootstrap draws. Without NumPy use `--skip-statistics`; that report remains partial. / NumPyなしでは`--skip-statistics`を使い、partialとして扱います。

The main figure supports the full 0–100% accuracy range in SVG and TeX, with space above the bars and a separate legend. Missing values remain NA. Display changes do not alter CSV values or statistical calculations.

