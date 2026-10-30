# Role 1 setup and local GPU smoke test

The proposal assigns Role 1 the training pipeline and integration. Qwen3-0.6B is the primary model. The setup script downloads its pinned snapshot into the repository-relative `models/Qwen3-0.6B` directory.

## Windows CUDA setup

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File HPML_WORK\windows\scripts\setup_issue7429_windows.ps1
```

Activate the environment if needed:

```powershell
.\.venv-issue7429\Scripts\Activate.ps1
```

Run the one-step LoRA + GRPO pipeline smoke test:

```powershell
python HPML_WORK\windows\scripts\role1_smoke.py
```

It checks model loading, Transformers generation, LoRA setup, reward integration, and one optimizer step. It is not a performance result and does not exercise vLLM. The native Windows weight-residency experiment is described in [ISSUE_7429_BENCHMARK.md](./ISSUE_7429_BENCHMARK.md).

## Linux GPU setup

For colocated vLLM and the actual issue reproduction, see the [Linux benchmark instructions](../../linux/docs/ISSUE_7429_BENCHMARK.md). Native Windows is not the supported target for vLLM. Use the same model revision, workload, and arguments for baseline and fixed runs.

## Role 1 next steps

1. Agree with the dataset/environment owner on shared dataset fields and tool-call format.
2. Replace the benchmark's synthetic CPU tool result with the shared chain-arithmetic workload for project-level measurements.
3. Run the unmodified Linux vLLM baseline and record software versions, synchronization/sleep counts, rollout/step latency, peak VRAM, and reward/accuracy.
4. Share the baseline and pinned environment with profiling and optimization owners.
