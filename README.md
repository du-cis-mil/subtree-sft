# Child-Subtree SFT

Code for **From Paper Selection to Child-Subtree Selection: Task-Aligned SFT for Hierarchical Scientific Literature Retrieval** (SoICT 2026).

This repository implements Paper-Selection and Child-Subtree supervised fine-tuning for hierarchical scientific literature retrieval. The experimental workflow is:

**Download TOMATO-Star → build training/evaluation trees → prepare two SFT datasets → train on GPUs → evaluate seven conditions → generate figures, tables, and statistics.**

## 1. Set up the repository

Use Python 3.11 and CUDA-capable GPUs. The training configuration uses **three GPUs**, bf16 full-parameter SFT of a 7B model, a 16,384-token sequence length, and DeepSpeed ZeRO-3 CPU offload. Provision GPU memory and host RAM for this configuration. Tree construction uses GPU embeddings and clustering; model serving also runs on a GPU.

Tested setup: Linux x86_64, CPython 3.11.18, NVIDIA driver 580, three RTX A6000 (48 GB) GPUs and 251 GiB host RAM. Training used up to about 41 GiB per GPU and about 215 GiB host RAM. Before installing, provide:

- a CUDA 12.6 toolkit (`nvcc`, cudart, CCCL, cuBLAS and cuRAND development files) and a C++ compiler; set `CUDA_HOME` to the toolkit. DeepSpeed builds its CPU-Adam extension at the start of training.
- a short `TMPDIR`, such as `/tmp`. Dataset conversion opens Unix sockets under `TMPDIR`, and a long path exceeds the 108-byte socket path limit.

To install exactly the package versions we tested, add `-c configs/constraints/<environment>.txt` to each install command ([details](docs/reproduction.md#1-environments-and-gpu-requirements)).

```bash
git clone https://github.com/du-cis-mil/subtree-sft.git
cd subtree-sft
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,eval,tree]' 'huggingface_hub>=0.34,<1'
```

Create separate training and SGLang environments as described in the [full experiment protocol](docs/reproduction.md#1-environments-and-gpu-requirements).

## 2. Download TOMATO-Star and the IR training data

Download both datasets from their official Hugging Face repositories:

- [TOMATO-Star](https://huggingface.co/datasets/ZonglinY/TOMATO-Star): raw training and test source records used to construct the trees, Child-Subtree supervision, and evaluation inputs.
- [TOMATO-Star-SFT-Data-R1D-32B](https://huggingface.co/datasets/ZonglinY/TOMATO-Star-SFT-Data-R1D-32B): the `IR/train.jsonl` conversations used for Paper-Selection SFT.

```bash
hf download ZonglinY/TOMATO-Star \
  --repo-type dataset --include train.jsonl test.jsonl \
  --local-dir data/raw/tomato_star

hf download ZonglinY/TOMATO-Star-SFT-Data-R1D-32B \
  --repo-type dataset --include 'IR/*' \
  --local-dir data/raw/sft
```

The resulting inputs are `data/raw/tomato_star/train.jsonl`, `data/raw/tomato_star/test.jsonl`, and `data/raw/sft/IR/train.jsonl`. See [data preparation](docs/data.md) for their roles and schema.

Download the backbone and public baseline into the Hugging Face cache:

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
hf download ZonglinY/MOOSE-Star-IR-R1D-7B
```

## 3. Build the training and evaluation trees

The training tree uses training inspirations. The evaluation tree uses a corpus assembled from training source papers, training inspirations, and held-out inspirations. Run these commands in order:

```bash
CUDA_VISIBLE_DEVICES=0 python -m soict_ir.build_tree \
  --sft_qa_jsonl data/raw/tomato_star/train.jsonl \
  --output_dir data/trees/training \
  --branching_factor 15 --use_medoid 1 --seed 42 \
  --kmeans_backend flash --device cuda:0 --flash_kmeans_device cuda:0

python -m soict_ir.corpus \
  --train-jsonl data/raw/tomato_star/train.jsonl \
  --test-jsonl data/raw/tomato_star/test.jsonl \
  --output-jsonl data/corpus/evaluation_papers.jsonl \
  --summary data/corpus/evaluation_summary.json

CUDA_VISIBLE_DEVICES=0 python -m soict_ir.build_tree \
  --paper_jsonl data/corpus/evaluation_papers.jsonl \
  --output_dir data/trees/evaluation \
  --branching_factor 15 --use_medoid 1 --seed 42 \
  --kmeans_backend flash --device cuda:0 --flash_kmeans_device cuda:0
```

Tree construction uses `allenai/specter2_base` embeddings, downloaded automatically on first use. The evaluation commands below use `--tree-profile rebuilt` for these newly constructed trees.

## 4. Prepare the two 10,000-example SFT datasets

```bash
python scripts/prepare_paper_selection.py \
  --input-jsonl data/raw/sft/IR/train.jsonl \
  --output-dir data/sft/paper_selection --max-examples 10000

python -m soict_ir.supervision \
  --input-jsonl data/raw/tomato_star/train.jsonl \
  --tree-dir data/trees/training \
  --output-dir data/sft/child_subtree --max-examples 10000

python scripts/audit_training_data.py \
  --source-train data/raw/tomato_star/train.jsonl \
  --paper-sft data/sft/paper_selection/train.jsonl \
  --child-sft data/sft/child_subtree/train.jsonl \
  --output outputs/training_source_audit.json
```

Each preparation command writes `train.jsonl` and the `dataset_info.json` required by LLaMA-Factory. The audit records source overlap, example counts, and the depth distribution for reporting.

## 5. Train both models on GPUs

After creating `.venv-train` with the [training dependencies](docs/reproduction.md#1-environments-and-gpu-requirements), run the two jobs sequentially from the repository root:

```bash
source .venv-train/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2 FORCE_TORCHRUN=1 NPROC_PER_NODE=3 \
  llamafactory-cli train configs/paper_selection.yaml
CUDA_VISIBLE_DEVICES=0,1,2 FORCE_TORCHRUN=1 NPROC_PER_NODE=3 \
  llamafactory-cli train configs/child_subtree.yaml
```

Both jobs start from `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and train for one epoch. Use the final checkpoints from `outputs/paper_selection/` and `outputs/child_subtree/`. The supplied three-GPU recipe uses step 3,334; check that the corresponding final checkpoint directories exist before serving them.

## 6. Run GPU evaluation

Split the test data and create a model map:

```bash
source .venv/bin/activate
python scripts/split_eval.py \
  --input-jsonl data/raw/tomato_star/test.jsonl --output-dir data/eval
cp configs/models.example.json configs/models.local.json
```

Set the checkpoint paths and server endpoints in `configs/models.local.json`. Start SGLang in a separate terminal; for the Paper-Selection model:

```bash
source .venv-serve/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path outputs/paper_selection/checkpoint-3334 \
  --served-model-name soict-repro --host 127.0.0.1 --port 30003 \
  --context-length 16384 --tp 1 --dtype bfloat16 \
  --chat-template outputs/paper_selection/checkpoint-3334/chat_template.jinja
```

Once the server is ready, evaluate both formats from the preparation/evaluation terminal:

```bash
python scripts/run_paper_experiments.py \
  --condition repro_child repro_original \
  --model-map configs/models.local.json --tree-profile rebuilt \
  --output-dir outputs/repro_run --execute
```

Follow the [complete seven-condition GPU schedule](docs/reproduction.md#5-serve-and-evaluate-all-seven-conditions) for the remaining models. It provides the server settings and executable evaluation commands for:

| Model | Name in the paper | Conditions | Evaluation context |
|---|---|---|---|
| Backbone | Base | `base_child` | 16,384 |
| Public MOOSE-Star IR | MOOSE-Star Pub | `public_child`, `public_original` | 16,384 / 32,768 |
| Paper-Selection SFT | MOOSE-Star Repro | `repro_child`, `repro_original` | 16,384 |
| Child-Subtree SFT | Child-Subtree SFT | `child_child`, `child_original` | 16,384 |

The `_child` conditions use the child-subtree-selection evaluation format and the `_original` conditions use the original MOOSE-Star IR format. Models can be served sequentially on the same GPU, or in parallel on different GPUs by changing `CUDA_VISIBLE_DEVICES`; each deployment has its own port. Each model's two prompt formats use the same checkpoint. Evaluation covers 1,658 held-out sources and decisions at depths D0–D4.

## 7. Generate the paper figures, tables, and statistics

After completing the five suite directories in the protocol:

```bash
python scripts/merge_paper_suites.py \
  --suite-dirs outputs/base_run outputs/public_child_run \
    outputs/public_original_run outputs/repro_run outputs/child_run \
  --output-dir outputs/paper_merged

python scripts/audit_corpus.py \
  --source-train data/raw/tomato_star/train.jsonl \
  --source-test data/raw/tomato_star/test.jsonl \
  --tree-dir data/trees/evaluation \
  --output outputs/corpus_audit.json \
  --tree-summary outputs/tree_granularity.json

python scripts/make_paper_report.py \
  --suite-dir outputs/paper_merged --output-dir outputs/report \
  --tree-summary outputs/tree_granularity.json \
  --training-audit outputs/training_source_audit.json \
  --corpus-audit outputs/corpus_audit.json
```

The report produces two figures, four tables, depth-wise accuracy, Top-k metrics, exact McNemar tests, and paired bootstrap confidence intervals. Files in `outputs/report/` that correspond to the paper:

| Paper | Files | Conditions |
|---|---|---|
| Fig. 1 (method overview) | `method_overview.*`, two concept PNGs | — |
| Fig. 2 (depth-wise Top-1) | `depthwise.csv`, `depthwise_figure.*` | `base_child`, `repro_original`, `child_child` |
| Table 1 (implementation) | `implementation.*`, `training_coverage.*` | — |
| Table 2 (tree granularity) | `tree_granularity.*` | — |
| Table 3 (public-model Top-k) | `public_topk_legacy.*` | `public_child` |
| Table 4 (format comparison) | `ablation.*` | all except `base_child` |
| Paired CIs and McNemar tests | `paired_tests.*`, `main_paired_statistics.json` | `child_child` vs `repro_original` |

Table 3 uses the paper's rank rule (`--profile historical`, the default), which also ranks decisions whose selection marker was not found. `public_topk_corrected.*` gives the same table with those decisions unranked. See the [output inventory](docs/manuscript_coverage.md) for all generated files.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and attribution.
