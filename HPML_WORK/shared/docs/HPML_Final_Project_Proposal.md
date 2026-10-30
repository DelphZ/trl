> [!NOTE]
> - Running headers, footers and page numbers removed: 3

HPML Final Project Proposal

High-Performance Machine Learning — Fall 2026

1. Project Title

Reducing Inter-Turn Overheads in Multi-Turn GRPO Training: Weight Reloading and Redundant Prefill

1. Team Members

Ann Zhang, Binyan Sun, Mingjing Yi, Yidan Wang, Yuyun Zhou

1. Goal/Objective

Group Relative Policy Optimization (GRPO) trains language models using groups of sampled responses and their relative rewards [1]. In agentic training, a rollout can contain multiple generation turns separated by tool execution. Our project studies the efficiency of these multi-turn rollouts when the training model and the vLLM inference engine share a single GPU. The project is motivated by TRL issue #7428 [2]. With colocated vLLM and sleep mode enabled, each generation call can end with level-2 sleep, discarding inference weights and KV cache. The next turn reloads the same policy despite no intervening optimizer update. The issue demonstrates this redundant behavior with a mock reproduction but does not quantify its GPU performance cost. We will investigate two questions: (1) How much do repeated weight synchronization and redundant prefill contribute to rollout and training-step latency? (2) Can rollout-level engine residency and cross-turn prefix caching reduce these costs without materially degrading task accuracy or training rewards? We aim to deliver a reproducible performance study and targeted TRL–vLLM improvements, aligned with the course directions of LLM fine-tuning and performance analysis.

1. Challenges

Weight and memory lifecycle. Keeping vLLM awake avoids repeated reloads, but inference memory must be released before training needs it. The implementation must distinguish CPU-only tool turns from optimizer updates and tools that require GPU memory. Cache correctness and capacity. Reusing KV requires matching token prefixes and unchanged policy weights. Cached states must be invalidated after weight updates. As conversation histories grow, limited cache capacity and eviction may reduce reuse or cause recomputation. Performance attribution. End-to-end latency includes synchronization, generation, tool execution, reward computation, and training. We must identify which components improve, while accounting for changes in generated lengths and actual tool-call counts.

1. Approach and Performance Optimization Techniques to be Used

Technique 1: Rollout-level sleep management. We will modify the TRL generation lifecycle so that vLLM remains awake throughout a multi-turn rollout with CPU tools. We will synchronize weights when needed, retain them between turns, and put the engine to sleep after the rollout, before training resumes. This removes redundant reloads without changing the GRPO objective. As a secondary comparison, we may evaluate level-1 sleep, which offloads weights to CPU and discards KV, rather than retaining weights on GPU [3].

Technique 2: Cross-turn prefix caching. Building on the awake policy, we will enable vLLM’s automatic prefix caching to reuse the unchanged conversation prefix instead of prefilling the entire history at every turn [4]. We will preserve consistent token sequences and ensure cache invalidation when policy weights change. The analysis will consider both avoided prefill work and the effects of cache capacity, eviction, and preemption. These techniques address complementary costs: weight movement and repeated context computation.

1. Implementation Details

Hardware and models. We plan to use one NVIDIA A6000 GPU on Columbia’s Insomnia cluster or NVIDIA A100 GPU on Runpod, with 40 or 80 GB of memory depending on availability. Qwen3-0.6B will be the primary model; Qwen3-1.7B will provide a model-size comparison if resources permit [5]. Training will use BF16 and LoRA through PEFT to keep the project feasible on a single GPU [6]. The actual GPU model and software versions will be recorded. Software and existing code. We will build on PyTorch, Hugging Face Transformers, TRL’s GRPOTrainer, and vLLM [7, 8]. The system will use use_vllm=True and vllm_mode="colocate". We will reuse the issue’s minimal reproduction to check synchronization behavior and extend the existing multi-turn tool loop and vLLM generation backend. An affected TRL revision will be pinned as the baseline, with targeted patches applied to that same revision. We will use PyTorch Profiler and, where available, Nsight Systems for performance analysis, and Weights & Biases for run tracking. Datasets and workloads. The primary workload will be a synthetic chain-arithmetic dataset paired with a CPU calculator environment. Each task contains a sequence of dependent operations, with the environment revealing the next operation after the preceding tool call. This provides controlled multi-turn interactions at different chain lengths, initially targeting 2, 4, and 8 tool calls. We will record actual generation turns and tool calls, since failed or prematurely terminated trajectories can differ from the intended chain length. Final-answer correctness will provide an automatically verifiable reward. We will additionally use a fixed subset of GSM8K, a public benchmark of grade-school mathematical word problems [9], with calculator access. Its training split will supply prompts for supplementary training runs, and held-out problems will support accuracy checks. GSM8K provides a natural-language task comparison, while the synthetic workload provides direct control over multi-turn behavior; GSM8K alone does not guarantee a particular number of tool calls. Performance assessment. We will compare the baseline, rollout-level sleep management, and sleep management plus prefix caching under matched hardware, models, and workload settings. Primary metrics will be rollout latency, training-step time, throughput, and GPU memory usage. Component timings and available vLLM cache statistics will help attribute savings to synchronization and prefill. Task accuracy and reward trends will check for material quality degradation. Detailed run counts and hyperparameters will be finalized after an initial feasibility run.

1. Demo Planned

We will demonstrate a short multi-turn calculator rollout using baseline and optimized implementations on the same inputs, showing interaction traces, synchronization and sleep counts, latency, and GPU memory behavior. Collected results will illustrate latency breakdowns, prefix reuse, and task-quality comparisons. A GitHub repository will contain patches, the workload generator, execution commands, and performance figures. We plan to share useful findings with the upstream TRL issue.

1. References

[1] Z. Shao et al., “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models,”

arXiv:2402.03300, 2024. https://arxiv.org/abs/2402.03300.

[2] ingambe, “Perf Issue: Colocated vLLM sleeps and reloads weights between the tool turns of one rollout,” Hugging

Face TRL, GitHub issue #7428, Sep. 28, 2026. https://github.com/huggingface/trl/issues/7428. Accessed Oct. 5, 2026.

[3] vLLM contributors, “Sleep Mode,” vLLM documentation. https://docs.vllm.ai/en/latest/features/sleep_mode/.

Accessed Oct. 5, 2026.

[4] vLLM contributors, “Automatic Prefix Caching,” vLLM documentation. https://docs.vllm.ai/en/latest/features/auto

matic_prefix_caching/. Accessed Oct. 5, 2026.

[5] Qwen Team, “Qwen3 Technical Report,” arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388.

[6] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv:2106.09685, 2021. https:

//arxiv.org/abs/2106.09685.

[7] Hugging Face, “GRPO Trainer,” TRL documentation. https://huggingface.co/docs/trl/grpo_trainer. Accessed Oct.

5, 2026.

[8] W. Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” in

Proceedings of SOSP, 2023. https://arxiv.org/abs/2309.06180.

[9] K. Cobbe et al., “Training Verifiers to Solve Math Word Problems,” arXiv:2110.14168, 2021. https://arxiv.org/abs/

2110.14168.
