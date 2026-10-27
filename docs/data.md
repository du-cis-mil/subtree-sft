# Data preparation

Obtain TOMATO-Star train/test records and the IR conversations from TOMATO-Star-SFT-Data-R1D-32B under their respective dataset terms. Model identifiers are specified in `configs/`.

Place the input files at:

```text
data/raw/tomato_star/train.jsonl
data/raw/tomato_star/test.jsonl
data/raw/sft/IR/train.jsonl
```

Raw records use `source_id`, `research_question`, `background_survey`, and `inspiration`. Inspiration annotations contain `found_title` / `supposed_title` and `found_abstract`. Title matching uses lowercase, whitespace-normalized exact matching.

Build separate training and evaluation trees following the [experiment protocol](reproduction.md). The evaluation corpus combines candidate papers from training sources and inspirations with held-out inspirations. The corpus audit records annotation matching, deduplication, and candidate counts.

For the original evaluation tree, `configs/historical_tree_identity.json` provides hashes of its structure, papers, and configuration files. Use `--tree-profile rebuilt` when evaluating a tree constructed from your own inputs.
