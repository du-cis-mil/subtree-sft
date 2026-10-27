# Reproduction protocol

Run the commands below from the repository root. Use separate environments for data preparation, training, and serving, and a new output directory for each experiment.

## 1. Create environments

Use Python 3.11 for the example setup. For data preparation and analysis:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
```

Install `.[tree]` only if rebuilding trees. It adds PyTorch, Transformers, scikit-learn, and Flash-KMeans. The historical trees used Flash-KMeans, not the sklearn alternative. SPECTER2-style embeddings in this implementation are CLS embeddings from `allenai/specter2_base`, without an additional adapter. Exact tree identity depends on corpus order, embedding files, clustering implementation, and hardware; seed 42 alone is insufficient. Compare against the bundled tree fingerprints or obtain the original trees from the authors.

For training, create a separate environment:

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-training.txt
python -m pip install -e .
```

Training dependencies follow LLaMA-Factory 0.9.3's [dependency bounds](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/requirements.txt) and [DeepSpeed extra](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/setup.py). Record the CUDA and package versions used for each run.

## 2. Obtain or rebuild both trees

Place data as described in [data.md](data.md). If rebuilding, use a new tree directory and run deliberately; these commands perform embedding inference and clustering:

```bash
python -m pip install -e '.[tree]'
python -m soict_ir.build_tree \
  --sft_qa_jsonl data/raw/tomato_star/train.jsonl \
  --output_dir data/trees/training \
  --branching_factor 15 --use_medoid 1 --seed 42 --kmeans_backend flash

python -m soict_ir.corpus \
  --train-jsonl data/raw/tomato_star/train.jsonl \
  --test-jsonl data/raw/tomato_star/test.jsonl \
  --output-jsonl data/corpus/evaluation_papers.jsonl \
  --summary data/corpus/evaluation_summary.json
python -m soict_ir.build_tree \
  --paper_jsonl data/corpus/evaluation_papers.jsonl \
  --output_dir data/trees/evaluation \
  --branching_factor 15 --use_medoid 1 --seed 42 --kmeans_backend flash
```

Use separate training and evaluation trees with seed42, branching factor15, maximum leaf depth5 and 768-dimensional embeddings. Rebuilding a different tree is a new realization of the method, not proof of exact reproduction. The historical builder can reuse embeddings in a directory, so use a fresh directory for a deliberate rebuild and archive its config and hashes.

## 3. Construct the two fixed-budget datasets

```bash
python -m soict_ir.supervision \
  --input-jsonl data/raw/tomato_star/train.jsonl \
  --tree-dir data/trees/training \
  --output-dir data/sft/child_subtree --max-examples 10000
python scripts/prepare_paper_selection.py \
  --input-jsonl data/raw/sft/IR/train.jsonl \
  --output-dir data/sft/paper_selection --max-examples 10000
python scripts/audit_training_data.py \
  --source-train data/raw/tomato_star/train.jsonl \
  --paper-sft data/sft/paper_selection/train.jsonl \
  --child-sft data/sft/child_subtree/train.jsonl \
  --output outputs/training_source_audit.json
```

Sources are processed in input-file order. Targets are deduplicated in annotation order. Internal nodes are collected in first encounter order across target paths, deduplicated by node ID. Candidates follow tree child order. The SFT label is the child containing the first target whose path passes through that node. All positive children remain in metadata. The cap applies after expansion; the last source record can be only partially included. Renaming a tree directory changes the informational `tree_name` metadata; retain the historical basename to obtain byte-identical training files with the exact original inputs.

Child-subtree assistant responses use the deterministic template in `build_assistant_response`.

## 4. Train from the same backbone

```bash
source .venv-train/bin/activate
FORCE_TORCHRUN=1 NPROC_PER_NODE=3 llamafactory-cli train configs/paper_selection.yaml
FORCE_TORCHRUN=1 NPROC_PER_NODE=3 llamafactory-cli train configs/child_subtree.yaml
```

Both configurations preserve full-parameter SFT, seed 42, one epoch, learning rate 1e-5, cosine schedule, 5% warmup, weight decay .01, bf16, batch 1 per device, accumulation 1, cutoff 16,384, max gradient norm .5, SDPA, and ZeRO-3 parameter/optimizer CPU offload. They start independently from the same backbone. Run from this directory so relative paths resolve correctly. Use three GPUs to preserve the historical effective batch and schedule. Changing world size changes optimization.

## 5. Evaluate local decisions at D0–D4

Create a separate serving environment using a compatible SGLang version and record the exact GPU/CUDA/package versions. The training requirements are not a serving environment. The following interface is an example; the exact commands and configurations should be checked against your own environment:

```bash
python -m sglang.launch_server \
  --model-path outputs/paper_selection/checkpoint-3334 \
  --served-model-name soict-repro --context-length 16384 \
  --host 127.0.0.1 --port 30003
```

`configs/models.example.json` declares the four model identities and five context deployments. Copy it to a local JSON file and update actual checkpoint paths, distinct served names and endpoints. The same checkpoint must be used for a model's two evaluation formats. Advertised `/models` names are checked before explicit execution; this verifies a name, not weight hashes or actual context capacity. Checkpoint paths and context lengths are operator declarations and remain labeled as such. Model servers are started manually. A server may be stopped by its operator after its conditions complete and the next model may then be served at its declared endpoint; simultaneous serving of all models is not required.

In the preparation/evaluation environment:

```bash
source .venv/bin/activate
python -m pip install -e '.[eval]'
python scripts/split_eval.py \
  --input-jsonl data/raw/tomato_star/test.jsonl --output-dir data/eval
```

### Seven-condition matrix

All commands below are **dry-runs**. They resolve actual settings and commands without loading weights, contacting servers, or creating run output directories. Add `--execute` only when deliberately running an evaluation with the corresponding server already available. Use a new output directory on every attempt.

```bash
python scripts/run_paper_experiments.py --condition base_child --model-map configs/models.example.json --output-dir outputs/base_child_run
python scripts/run_paper_experiments.py --condition public_child --model-map configs/models.example.json --output-dir outputs/public_child_run
python scripts/run_paper_experiments.py --condition public_original --model-map configs/models.example.json --output-dir outputs/public_original_run
python scripts/run_paper_experiments.py --condition repro_child --model-map configs/models.example.json --output-dir outputs/repro_child_run
python scripts/run_paper_experiments.py --condition repro_original --model-map configs/models.example.json --output-dir outputs/repro_original_run
python scripts/run_paper_experiments.py --condition child_child --model-map configs/models.example.json --output-dir outputs/child_child_run
python scripts/run_paper_experiments.py --condition child_original --model-map configs/models.example.json --output-dir outputs/child_original_run
```

Use `--condition all` for one plan covering all seven conditions. `--plan-output outputs/plan.json` saves a new plan file. `--inspect-inputs` additionally hashes local data/tree files on CPU; it still makes no requests. Resolved per-evaluation commands reference source/run-spec files written by the explicit suite executor, so execute through the runner rather than copying an unprepared inner command from a dry-run JSON.


Serve models sequentially if needed, then merge validated condition suites:

```bash
python scripts/merge_paper_suites.py --suite-dirs outputs/base_child_run outputs/public_child_run outputs/public_original_run outputs/repro_child_run outputs/repro_original_run outputs/child_child_run outputs/child_original_run --output-dir outputs/paper_merged
python scripts/audit_corpus.py --source-train data/raw/tomato_star/train.jsonl --source-test data/raw/tomato_star/test.jsonl --tree-dir data/trees/evaluation --output outputs/corpus_audit.json --tree-summary outputs/tree_granularity.json
python scripts/make_paper_report.py --suite-dir outputs/paper_merged --output-dir outputs/report --tree-summary outputs/tree_granularity.json --training-audit outputs/training_source_audit.json --corpus-audit outputs/corpus_audit.json
```

Use the final epoch checkpoint-3334 for each independently trained model; do not select on held-out accuracy. Both training datasets contain exactly 10,000 examples. Training and evaluation trees are separate. Historical tree sizes or source coverage observations are not promises about rebuilt artifacts.

Historical tree mode checks the structure/papers/config hashes in configs/historical_tree_identity.json. A deliberately rebuilt tree requires explicit --tree-profile rebuilt; the code never falls back silently. Toy mode requires a toy tree and cannot establish paper completeness.

The report requires valid source sets, five depths, protocol/model consistency, integer corpus count relationships, fixed 10,000-example budgets and the pinned 159-field output schema. Null/bool/float budgets and impossible corpus counts are errors in strict and partial mode. Consistent incomplete mapping/depth coverage is partial with explicit missing reasons. A subset or mutation of the inventory is always rejected.

The report generates four tables, two figures, corpus counts, model differences, macro differences, exact McNemar tests, and paired bootstrap intervals from the experiment inputs. Missing values are marked as `NA`. `--skip-statistics` produces a partial report; install the analysis extra to compute statistics.

Bootstrap uses 5,000 paired draws at seed 42, one PCG64 generator advanced through D0–D4, sorted source IDs, and linear percentile quantiles. The intervals quantify source-sampling uncertainty.

Corrected Top-k removes ranks on failed-format parents, takes the best valid rank across a source's parents, and keeps all evaluation sources in its denominator. Mean/median are conditional on having a valid rank. Top-1 uses actual correctness rather than candidate-order ties. Local depths are evaluated independently, not end-to-end traversal.


The main figure supports the full 0–100% accuracy range in SVG and TeX, with space above the bars and a separate legend. Missing values remain NA. Display changes do not alter CSV values or statistical calculations.

