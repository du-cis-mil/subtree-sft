# Download and prepare the data

## Required datasets

| Dataset | Files to download | Role in the experiments |
|---|---|---|
| [TOMATO-Star](https://huggingface.co/datasets/ZonglinY/TOMATO-Star) | `train.jsonl`, `test.jsonl` | Raw source papers, inspirations, tree construction, Child-Subtree supervision, held-out evaluation |
| [TOMATO-Star-SFT-Data-R1D-32B](https://huggingface.co/datasets/ZonglinY/TOMATO-Star-SFT-Data-R1D-32B) | `IR/train.jsonl`, `IR/dataset_info.json` | Released IR conversations from which the Paper-Selection training prefix is selected |

Use the `hf` CLI from [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/cli) in the preparation environment:

```bash
source .venv/bin/activate
hf download ZonglinY/TOMATO-Star \
  --repo-type dataset --include train.jsonl test.jsonl \
  --local-dir data/raw/tomato_star
hf download ZonglinY/TOMATO-Star-SFT-Data-R1D-32B \
  --repo-type dataset --include 'IR/*' \
  --local-dir data/raw/sft
```

The raw TOMATO-Star training and test splits contain 107,059 and 1,658 source records respectively. Keep the two datasets distinct: raw TOMATO-Star records provide scientific context and inspiration annotations; the SFT dataset provides conversations. Follow the dataset terms and attribution provided by their authors.

## Expected layout

```text
data/
  raw/
    tomato_star/
      train.jsonl
      test.jsonl
    sft/
      IR/
        train.jsonl
        dataset_info.json
  corpus/                     # Created by soict_ir.corpus
  trees/
    training/                 # Created from training inspirations
    evaluation/               # Created from the evaluation candidate corpus
  sft/
    paper_selection/          # 10,000 IR conversations + dataset_info.json
    child_subtree/            # 10,000 child-subtree examples + dataset_info.json
  eval/                       # One JSON file per held-out source
```

## Input schema and processing

Raw records use `source_id`, `research_question`, `background_survey`, and `inspiration`. Inspiration annotations contain `found_title` / `supposed_title` and `found_abstract`. Title matching uses lowercase, whitespace-normalized exact matching.

The training tree contains inspiration papers from the training split. The evaluation corpus combines training source papers, training inspirations, and held-out inspirations. Build these trees separately using the [GPU tree construction commands](reproduction.md#2-download-inputs-and-build-the-trees).

Paper-Selection preparation takes the first 10,000 nonempty records of `IR/train.jsonl` in file order. Child-Subtree preparation expands the training sources into node decisions and takes 10,000 examples after expansion. Both write LLaMA-Factory-compatible dataset metadata. `audit_training_data.py` checks their source correspondence and records overlap and depth counts for the report.

Split the held-out raw records for evaluation:

```bash
python scripts/split_eval.py \
  --input-jsonl data/raw/tomato_star/test.jsonl --output-dir data/eval
```

For a newly constructed tree, pass `--tree-profile rebuilt` to the evaluation runner. For the original tree, `--tree-profile historical` checks its structure, papers, and configuration against `configs/historical_tree_identity.json`.
