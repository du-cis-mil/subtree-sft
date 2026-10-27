# GPU experiment protocol

Run commands from the repository root. The workflow uses separate environments for preparation/evaluation, SFT training, and SGLang serving. Use new output directories for each experiment.

## 1. Environments and GPU requirements

The SFT recipe uses three CUDA GPUs, bf16, full-parameter training of DeepSeek-R1-Distill-Qwen-7B, a sequence cutoff of 16,384, and ZeRO-3 parameter/optimizer CPU offload. Host RAM is used for offloaded parameters and optimizer states. A compatible CUDA toolkit and C++ compiler are required for DeepSpeed extensions; configure `CUDA_HOME` to the toolkit matching the installed PyTorch build.

### Preparation, tree construction, and evaluation client

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,eval,tree]' 'huggingface_hub>=0.34,<1'
```

The tree extra installs the embedding and clustering dependencies. `build_tree` uses CLS embeddings from `allenai/specter2_base` and Flash-KMeans with medoid representatives. Keep this environment active for dataset preparation, evaluation clients, and aggregation.

### SFT training

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-training.txt
python -m pip install -e .
```

The training requirements specify LLaMA-Factory 0.9.3, PyTorch 2.7.1, Transformers 4.52.4, and DeepSpeed 0.16.9. Keep the installed CUDA/PyTorch combination compatible with the driver and extension build tools.

### SGLang serving

```bash
python3.11 -m venv .venv-serve
source .venv-serve/bin/activate
python -m pip install 'sglang[all]==0.4.9.post6'
```

SGLang 0.4.9.post6 was used for the scaled execution check. Resolve its dependencies for the host CUDA platform using the [SGLang documentation](https://docs.sglang.ai/). Record the resolved versions for your experiment. GPU memory requirements depend on context length, tensor parallelism, and concurrent requests; tune deployment capacity to your hardware while keeping the evaluation context and model declarations consistent.

## 2. Download inputs and build the trees

Activate `.venv` and follow [data.md](data.md) to download TOMATO-Star and the released IR conversations. Also download the backbone and public baseline:

```bash
source .venv/bin/activate
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
hf download ZonglinY/MOOSE-Star-IR-R1D-7B
```

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

Trees use branching factor 15, medoid representatives, seed 42, maximum leaf depth 5, and 768-dimensional embeddings. The commands select GPU 0 for both embedding and clustering. Tree identity depends on input ordering, embeddings, clustering, and hardware; preserve each tree's configuration and hashes with the run.

## 3. Prepare training and evaluation data

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

Paper-Selection preserves the ordered prefix of the released IR conversations. Child-Subtree expands source records in file order, deduplicates targets in annotation order, and visits internal nodes in first encounter order. Candidate order follows tree child order. The label selects the child containing the first target whose path passes through the current node; metadata retains all positive children. The 10,000-example limit is applied after expansion.

Prepare the held-out evaluation inputs:

```bash
python scripts/split_eval.py \
  --input-jsonl data/raw/tomato_star/test.jsonl --output-dir data/eval
```

## 4. Train the two SFT models

```bash
source .venv-train/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2 FORCE_TORCHRUN=1 NPROC_PER_NODE=3 \
  llamafactory-cli train configs/paper_selection.yaml
CUDA_VISIBLE_DEVICES=0,1,2 FORCE_TORCHRUN=1 NPROC_PER_NODE=3 \
  llamafactory-cli train configs/child_subtree.yaml
```

Run these jobs sequentially. Both start from `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, with 10,000 examples, seed 42, one epoch, learning rate 1e-5, cosine scheduling, 5% warmup, weight decay 0.01, per-device batch size 1, accumulation 1, gradient clipping 0.5, bf16, SDPA, gradient checkpointing, and ZeRO-3 CPU offload. Three GPUs give an effective batch size of 3. Changing GPU count or accumulation changes the optimization schedule.

Use each model's final epoch checkpoint. The supplied recipe uses step 3,334; the examples below expect `outputs/paper_selection/checkpoint-3334` and `outputs/child_subtree/checkpoint-3334`. Confirm the saved final checkpoint paths, including tokenizer files and `chat_template.jinja`, and update the model map and serving commands together if paths differ.

## 5. Serve and evaluate all seven conditions

Create a local model map in the preparation/evaluation terminal:

```bash
source .venv/bin/activate
cp configs/models.example.json configs/models.local.json
```

The five deployments below match its served names, context lengths, and ports. Models can be run sequentially on GPU 0. For each deployment, start the server in a separate terminal, wait until it is ready, then run the corresponding evaluation command in the `.venv` terminal. After that evaluation finishes, stop that server with Ctrl-C before starting the next deployment on the same GPU.

The commands use `--tree-profile rebuilt` for trees constructed in step 2. Use `historical` only with the exact tree described in `configs/historical_tree_identity.json`. Evaluation is triggered by `--execute`; omitting it prints the execution plan.

### A. Backbone: Child-Subtree

Server:

```bash
source .venv-serve/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --served-model-name soict-base --host 127.0.0.1 --port 30000 \
  --context-length 16384 --tp 1 --dtype bfloat16
```

Evaluation:

```bash
python scripts/run_paper_experiments.py \
  --condition base_child --model-map configs/models.local.json \
  --tree-profile rebuilt --output-dir outputs/base_run --execute
```

### B. Public MOOSE-Star IR: Child-Subtree

Server:

```bash
source .venv-serve/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path ZonglinY/MOOSE-Star-IR-R1D-7B \
  --served-model-name soict-public --host 127.0.0.1 --port 30001 \
  --context-length 16384 --tp 1 --dtype bfloat16
```

Evaluation:

```bash
python scripts/run_paper_experiments.py \
  --condition public_child --model-map configs/models.local.json \
  --tree-profile rebuilt --output-dir outputs/public_child_run --execute
```

### C. Public MOOSE-Star IR: Original IR

Serve the same checkpoint at the Original IR context length:

```bash
source .venv-serve/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path ZonglinY/MOOSE-Star-IR-R1D-7B \
  --served-model-name soict-public --host 127.0.0.1 --port 30002 \
  --context-length 32768 --tp 1 --dtype bfloat16
```

Evaluation:

```bash
python scripts/run_paper_experiments.py \
  --condition public_original --model-map configs/models.local.json \
  --tree-profile rebuilt --output-dir outputs/public_original_run --execute
```

### D. Paper-Selection SFT: both formats

Server:

```bash
source .venv-serve/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path outputs/paper_selection/checkpoint-3334 \
  --served-model-name soict-repro --host 127.0.0.1 --port 30003 \
  --context-length 16384 --tp 1 --dtype bfloat16 \
  --chat-template outputs/paper_selection/checkpoint-3334/chat_template.jinja
```

Evaluation:

```bash
python scripts/run_paper_experiments.py \
  --condition repro_child repro_original \
  --model-map configs/models.local.json --tree-profile rebuilt \
  --output-dir outputs/repro_run --execute
```

### E. Child-Subtree SFT: both formats

Server:

```bash
source .venv-serve/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path outputs/child_subtree/checkpoint-3334 \
  --served-model-name soict-child --host 127.0.0.1 --port 30004 \
  --context-length 16384 --tp 1 --dtype bfloat16 \
  --chat-template outputs/child_subtree/checkpoint-3334/chat_template.jinja
```

Evaluation:

```bash
python scripts/run_paper_experiments.py \
  --condition child_child child_original \
  --model-map configs/models.local.json --tree-profile rebuilt \
  --output-dir outputs/child_run --execute
```

The runner verifies the server's advertised model name against the map. Keep each model's checkpoint identical across its two formats. `configs/experiments.json` defines generation limits (512 for Base, 2,048 for fine-tuned models), top-30 log probabilities, and retry/worker settings. Context length and generation length are separate settings. Evaluation measures local decisions independently at D0–D4.

## 6. Aggregate experiments and generate figures and tables

Activate `.venv` after completing the five suite directories in the protocol:

```bash
source .venv/bin/activate
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

The report produces two figures, four tables, depth-wise accuracy, Top-k metrics, exact McNemar tests, and paired bootstrap confidence intervals. See the [output inventory](manuscript_coverage.md) for the generated files.

Reports require all seven conditions, five-depth coverage of the same 1,658 sources, consistent model/tree settings, valid corpus counts, and both 10,000-example training audits. Use `--allow-partial` to inspect an unfinished run, with missing fields marked as `NA`.

Top-k uses the best valid positive rank across a source's parent decisions; failed-format parents contribute no rank, and all sources remain in the denominator. Mean/median ranks are conditional on a valid rank. Top-1 uses source-level correctness.

Paired bootstrap uses 5,000 draws, seed 42, a PCG64 generator advanced through D0–D4, sorted source IDs, and linear percentile quantiles. Confidence intervals quantify source-sampling uncertainty. Exact McNemar tests compare paired source outcomes.
