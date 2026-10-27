# Output coverage / 出力台帳

configs/output_schema.json is schema2: a pinned inventory of all 159 output fields with the same IDs, computation paths, locations, precision/types and two-figure/four-table labels as the reviewed paper inventory. Historical expected numbers are not distributed. Changing or reducing this inventory is rejected even in partial mode.

固定159項目は利用者の実入力から計算し、computed_value / unverified / method_descriptionで区別します。原稿値は非同梱で、未比較を一致と表示しません。旧schema1の過去比較機能は別経路として保持しています。

| Label | Generated files |
|---|---|
| fig:method_overview | method_overview.html/.tex and two concept PNGs |
| fig:depthwise_top1 | depthwise.csv, depthwise_figure.svg/.tex (three series, six categories) |
| tab:implementation | implementation.csv/.tex, training_coverage, training_depth_distribution |
| tab:tree_granularity | tree_granularity.csv/.tex |
| tab:sample_level | public_topk_corrected.csv/.tex/.json |
| tab:prompt_ablation | ablation.csv/.tex (six rows) |

Body outputs include corpus_counts, parent_counts, text_comparisons, paired_tests, main_paired_statistics, ci_comparison, computed_values and per-field coverage. CI comparison references state not provided. --skip-statistics marks missing computations; --allow-partial never bypasses malformed inputs. See README and reproduction protocol for completion definitions.
