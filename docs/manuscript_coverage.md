# Figures, tables, and report fields

`configs/output_schema.json` defines 159 report fields, including their identifiers, computation paths, locations, types, and precision. The report validates this inventory and calculates each field from the experiment inputs. Fields distinguish computed values, method descriptions, and values requiring further inputs.

| Label | Generated files |
|---|---|
| fig:method_overview | method_overview.html/.tex and two concept PNGs |
| fig:depthwise_top1 | depthwise.csv, depthwise_figure.svg/.tex (three series, six categories) |
| tab:implementation | implementation.csv/.tex, training_coverage, training_depth_distribution |
| tab:tree_granularity | tree_granularity.csv/.tex |
| tab:sample_level | public_topk_corrected.csv/.tex/.json |
| tab:prompt_ablation | ablation.csv/.tex (six rows) |

Body outputs include corpus_counts, parent_counts, text_comparisons, paired_tests, main_paired_statistics, ci_comparison, computed_values and per-field coverage. `--skip-statistics` marks the report as partial. `--allow-partial` reports missing inputs while retaining schema and input validation. See the [experiment protocol](reproduction.md) for the full workflow.
