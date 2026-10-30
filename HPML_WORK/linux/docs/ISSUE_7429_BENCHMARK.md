# Linux CUDA/vLLM benchmark: issue #7428 / PR #7429

GitHub #7428 reports redundant weight synchronization between multi-turn generation calls in one rollout when colocated vLLM sleep mode is enabled. #7429 is the proposed fix.

The upstream issue's minimal logical reproduction uses CPU and a mocked vLLM; that is the upstream author's unit-test-style reproduction, not this project's experiment. Our Insomnia workflow requires a GPU: Qwen3 inference, the vLLM benchmark, and GRPO model forward/backward training all run on CUDA, and both Python programs fail if their trainer is not assigned to a CUDA device. CPU work is limited to host orchestration, reward calculation, and the toy calculator/tool-result simulation between GPU generation turns; no model training or generation falls back to CPU. The Linux benchmark requires real vLLM and the proposal's Qwen3-0.6B model, and measures synchronization/sleep counts, sync/turn/rollout latency, and peak GPU memory. The native-Windows emulation and visualization steps are in the [Windows benchmark guide](../../windows/docs/ISSUE_7429_BENCHMARK.md).

## Setup and model download

From the repository root on Linux x86-64 with a supported NVIDIA driver, load the cluster's Anaconda module and create a persistent Python 3.12 environment. The setup script uses that interpreter to create its package virtual environment:

```bash
module load anaconda/2023.09
conda create --prefix "$HOME/.conda/envs/trl-issue7429" python=3.12 -y
PYTHON="$HOME/.conda/envs/trl-issue7429/bin/python" \
  bash HPML_WORK/linux/scripts/setup_issue7429_linux.sh
source .venv-issue7429/bin/activate
```

The setup creates a clean virtual environment, installs this checkout with its PEFT/vLLM extras plus plotting dependencies, verifies CUDA/vLLM, and downloads the pinned Qwen3-0.6B snapshot into the repository-relative `models/Qwen3-0.6B` directory. If using a different Conda prefix, set `PYTHON` to that environment's Python executable.

```bash
PYTHON=/path/to/conda/env/bin/python bash HPML_WORK/linux/scripts/setup_issue7429_linux.sh
```

To download separately:

```bash
python HPML_WORK/shared/scripts/download_qwen3.py
```

## Run

This script measures **only the checked-out TRL source**; it does not switch between or emulate the affected and fixed implementations. Run it once in the affected baseline checkout and once in the checkout containing the fix. Each run's JSON records the observed sync/sleep counts and source commit.

```bash
python HPML_WORK/linux/scripts/issue7429_live_benchmark.py \
  --label baseline \
  --turns 2 \
  --iterations 5 \
  --dtype bfloat16 \
  --vllm-gpu-memory-utilization 0.3 \
  --output HPML_WORK/linux/results/baseline.json
```

Repeat the same command in the fixed checkout, changing only `--label fixed` and the output filename to `fixed.json`. The model defaults to `models/Qwen3-0.6B` under the repository root; the model path, revision, GPU, and all benchmark settings must match between runs. Lower vLLM memory utilization if colocated allocation leaves too little room, and use the same value in both runs.

For two turns, the expected observations from the **two separate source revisions** are:

| Separate run against | Weight syncs per rollout | Sleep calls inside rollout | Sleep after rollout |
|---|---:|---:|---:|
| Affected baseline checkout | 2 | 2 | 0 |
| Fixed checkout | 1 | 0 | 1 |

The table is not two policies exercised by one benchmark invocation: each row comes from running the same script against a different checkout. Zero sleeps **inside** the fixed rollout does not mean sleep mode was disabled; the engine sleeps once after the complete rollout, before scoring/training needs its GPU memory. `behavior` in each JSON should be `buggy` for the affected checkout and `fixed` for the fixed checkout; `source_commit` identifies exactly what was measured. The harness stops at the pre-scoring boundary so the issue-specific measurement isolates inter-turn behavior and skips an optimizer step. For end-to-end project metrics, plug in the shared chain-arithmetic dataset/environment and record accuracy/reward as well.

## Submit a full training and benchmark job to Columbia Insomnia

The Slurm job runs two phases on one allocated GPU:

1. The real colocated-vLLM issue benchmark above, saving a JSON report that the existing visualizer can read.
2. A configurable Qwen3-0.6B LoRA/GRPO training run on a small synthetic arithmetic dataset, saving the final adapter and separate training timing/memory metrics.

The training run is GPU LoRA/GRPO training on the allocated Insomnia GPU, not CPU training. It is a runnable integration workload, not the project's final multi-turn calculator/GSM8K experiment. It currently trains for 10 optimizer steps by default; adjust the workload after the team dataset and environment are ready.

Log in to Insomnia using the Columbia-provided hostname and your UNI, then go to the repository root. Prepare the environment/model once in a cluster allocation where CUDA is visible to PyTorch (the setup script verifies GPU availability). The batch script loads `anaconda/2023.09` and activates `$HOME/.conda/envs/trl-issue7429` before activating the package virtual environment. Create the Conda environment once, then run setup:

```bash
module load anaconda/2023.09
conda create --prefix "$HOME/.conda/envs/trl-issue7429" python=3.12 -y
PYTHON="$HOME/.conda/envs/trl-issue7429/bin/python" \
  bash HPML_WORK/linux/scripts/setup_issue7429_linux.sh
```

Inspect the cluster's current GPU partitions and constraints, then adjust the `#SBATCH` account, partition, GPU, memory, and time directives in [`insomnia_grpo_benchmark.sbatch`](../scripts/insomnia_grpo_benchmark.sbatch) if required. Submit from the repository root so Slurm can create its log files in the existing results directory:

```bash
sbatch HPML_WORK/linux/scripts/insomnia_grpo_benchmark.sbatch
squeue -u "$USER"
```

By default, the job writes:

- `HPML_WORK/linux/results/insomnia-JOBID-benchmark.json` — issue-specific vLLM benchmark, directly consumable by the visualizer.
- `HPML_WORK/linux/results/insomnia-JOBID-benchmark.png` — visualization generated from that JSON.
- `HPML_WORK/linux/results/insomnia-JOBID-training.json` — training duration, step timings, loss, and peak memory.
- `HPML_WORK/linux/results/insomnia-JOBID-training/final_adapter/` — trained LoRA adapter.
- `HPML_WORK/linux/results/insomnia-JOBID.out` and `.err` — Slurm stdout/stderr.

To compare baseline and fixed revisions, use two separate working trees/checkouts on Insomnia so a queued job cannot start after its branch has been changed. Install an environment in each checkout and download/share the model as appropriate. Submit one benchmark-only job from each checkout:

```bash
HPML_RUN_TRAINING=0 HPML_BENCHMARK_LABEL=baseline \
  HPML_BENCHMARK_OUTPUT=HPML_WORK/linux/results/baseline.json \
  HPML_BENCHMARK_PLOT_OUTPUT=HPML_WORK/linux/results/baseline.png \
  sbatch --export=ALL HPML_WORK/linux/scripts/insomnia_grpo_benchmark.sbatch
HPML_RUN_TRAINING=0 HPML_BENCHMARK_LABEL=fixed \
  HPML_BENCHMARK_OUTPUT=HPML_WORK/linux/results/fixed.json \
  HPML_BENCHMARK_PLOT_OUTPUT=HPML_WORK/linux/results/fixed.png \
  sbatch --export=ALL HPML_WORK/linux/scripts/insomnia_grpo_benchmark.sbatch
python HPML_WORK/shared/scripts/visualize_benchmark.py \
  --input HPML_WORK/linux/results/baseline.json HPML_WORK/linux/results/fixed.json \
  --output HPML_WORK/linux/results/insomnia-comparison.png
```

Optional job settings can be supplied as exported environment variables before `sbatch`: `HPML_VENV`, `HPML_MODEL`, `HPML_DTYPE`, `HPML_TURNS`, `HPML_BENCHMARK_ITERATIONS`, `HPML_RUN_TRAINING`, `HPML_TRAIN_STEPS`, `HPML_VLLM_GPU_MEMORY_UTILIZATION`, `HPML_BENCHMARK_LABEL`, `HPML_BENCHMARK_OUTPUT`, `HPML_BENCHMARK_PLOT_OUTPUT`, `HPML_TRAIN_OUTPUT_DIR`, and `HPML_TRAIN_METRICS_OUTPUT`. Slurm exports the submitting shell's environment with `--export=ALL`. Ensure the model, GPU allocation, and benchmark settings match; the source commit should differ only as intended.

## Visualize and compare

Save separate baseline and fixed JSON results using `--output`, then create one comparison plot:

```bash
python HPML_WORK/shared/scripts/visualize_benchmark.py \
  --input HPML_WORK/linux/results/baseline.json HPML_WORK/linux/results/fixed.json \
  --output HPML_WORK/linux/results/comparison.png
```

The figure compares rollout time, weight transfer time, per-turn generation latency, and peak allocated VRAM. Keep hardware, model revision, software, workload, generated token counts, and all CLI arguments matched; repeat runs and report medians/spread.
