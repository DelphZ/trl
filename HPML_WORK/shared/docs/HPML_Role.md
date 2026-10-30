Roles:

| 内容 | 具体任务 | 目标产出 | 成员 |
| --- | --- | --- | --- |
| 训练Pipeline与 集成 | 配置单卡 TRL + vLLM colocate；接入 Qwen3 和 LoRA；跑通多轮 GRPO baseline；固定依赖和配置 ；合并其他人的改动 | 可运行的 baseline； 环境说明；统一运行 命令 | Yuyun |
| Dataset、工具 环境与质量评 测 | 构造链式算术任务和 CPU calculator；支持不同链长； 定义正确答案和 reward； 接入 GSM8K 子集；检查 accuracy 和 reward | 工具环境；质量评测 脚本 | Yidan |
| Profiling 与性 能评测 | 测 rollout / step latency、 throughput、显存；使用 Profiler / Nsight；组织对照 结果和图表 | 时间分解；性能对比 图；Demo 性能展示 | Ann |
| 权重重载优化 | 复现 issue #7428；追踪 sync / sleep / wake 调用； 把 sleep 移到完整 rollout 结束后；验证下一次训练更 新后能同步新权重 | 生命周期补丁；同步 次数对比；正确性检 查 | Binyan |
| Prefix Caching 优化 | 在 awake 版本上启用缓存 ；检查跨轮 token 前缀；验 证权重更新后缓存失效；分 析命中、驱逐和 preemption | 缓存配置或补丁； prefill 与缓存行为分 析 | Mingjing |
