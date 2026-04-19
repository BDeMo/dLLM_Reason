# Plan: Strategy Search (B1 / SS) 5D 设计

**日期**：2026-04-16（首版）
**状态**：superseded（被 [2026-04-19_replan_next_phase.zh.md](2026-04-19_replan_next_phase.zh.md) 取代）
**前置上下文**：[2026-04-15_a_axis_discovery_phase.zh.md](2026-04-15_a_axis_discovery_phase.zh.md) 完工，A 轴全部 SUPPORTED + H3 86.67%。下一步想从"每个 axis 独立实验"升级到"**per-prompt 多维联合搜索**"。
**继任**：[2026-04-19_ss_run_postmortem.zh.md](2026-04-19_ss_run_postmortem.zh.md) → [2026-04-19_replan_next_phase.zh.md](2026-04-19_replan_next_phase.zh.md)

## 目标

对 n=60 fail + 49 ok 的每条 prompt，在 5D 空间里穷搜"每条 prompt 的最佳策略"，产出 `(prompt, best_strategy)` pair，喂下阶段 SFT distill。

这是 paper 的核心实验 —— 证明 **per-prompt 策略 > 任何全局最优单一策略**。

## 决策 / 方法

### 5D 搜索空间（default pruned）

| 维度 | 值集 | 理由 |
|---|---|---|
| `block_length` | {16, 32, 64} | 砍了 bl=8（A4 里 bl8 broken 多） |
| `template_name` | {baseline, cot_plain, cot_step, answer_marker, step_by_step_prompt} | A5 四选一 + 新增 step_by_step_prompt |
| `template_position` | **{prefix, suffix_scaffold, mid_anchor, none}** | **Diffusion LM 独有新维度**（paper 核心 claim） |
| `gen_length` | {128, 160, 192} | 砍了 g=64/96/256（A6 里弱档）|
| `temperature` | {0.0, 0.3, 0.7} | 砍了 T=1.0（H3 里 T=0.7 已够） |

合法 (bl, gen) 约束：`gen % bl == 0` → 8 pairs。
去重：`position=none` 强制 `tmpl=baseline`（避免冗余）。
pass@N：T=0 → N=1，T>0 → N=4。

### Budget 预估（ex-ante）

- Configs/prompt: 384
- Samples/prompt: 128 (T=0) + 1024 (T>0) = **1152**
- Calls 全 109 prompts: ~125k
- 单 GPU @ 1.5s/call: ~52 小时
- 8 GPU 分片 @ 1.5s/call: ~6.5 小时

**（这个预估后来证明严重偏低，见 2026-04-19 postmortem）**

### 工具链

- `scripts/validate/strategy_search.py` —— 5D 搜索 + 4 种 winner 挑选（cheapest / shortest / most_reliable / deterministic）
- `scripts/validate/run_ss_shards.sh` —— 多 GPU 分片 orchestrator（共用 run_dir，prompt 切片）
- `scripts/validate/ss_analyze.py` —— 跑完后的 paper-ready 分析（7 段报告 + FAIL18 逐条表）
- `scripts/validate/ss_to_sft.py` —— winners → SFT JSONL
- `docs/archive/ss_oracle_prior.zh.md` —— ex-ante oracle rate 先验（中位 55/60 = 91.67%）

## 预期输出

- `runs/validation/strategy_search_<ts>/` 含 109 条 `per_prompt/*.json` + `winners.json` + `summary.json`
- `analysis_report.md` 定 template_position novelty 的 paper claim
- `sft_train.jsonl / sft_val.jsonl` 喂 distillation baseline

## Updates

- **2026-04-16**: Orchestrator + strategy_search CLI 上 dev（commits `875df4a`, `7075a2b`, `cba1bad`）
- **2026-04-16**: Oracle prior 归档 + ablation_index 加 Setting 段
- **2026-04-17 04:26**: SS 首次 full run 启动（RUN_DIR `strategy_search_20260417_042620`）
- **2026-04-17 ~晚**: 用户发现 client log 空 → Python stdout 全缓冲问题，commit `3ac3ac1` 加 `PYTHONUNBUFFERED=1`
- **2026-04-19**: 确认跑 ~48h 只完成 15/109 prompt（13.8% 的 config 数）。详见 postmortem。

## Retrospective

### 跟预期不符的地方

| 预期 | 实际 | 原因 |
|---|---|---|
| 6.5h 全部完成（8 GPU）| ~85h per shard 估算，14 天全 run | 单 call 5s 估算偏低 |
| curl 5.5s = 实际 per-call 时间 | 实际 gsm8k + gen=192 inpaint 约 20-30s | seq² attention cost，真实 prompt ≫ test prompt |
| GPU util > 90% | 长期 0 偶尔 18% | serial HTTP + 长尾 forward |

### 教训

1. **Budget 估算必须用真实 prompt 测，不能用 "What is 2+2?" 这种 toy prompt** —— attention O(seq²) 放大效应很大
2. **Ex-ante 用 5.5s/call 假设是严重低估**。正确做法：挑 3 条真实 fail prompt × 2-3 种代表 config，端到端实测 per-sample 时间，乘 total samples
3. **GPU util 不满说明架构就不行**。SS 单 shard 一个 server 顺序打 request，server 处理时 client 空转 + client 处理时 server 空转。需要 concurrent pipeline 或 server batching
4. **5D × 全 109 条 × 1152 samples 过激进**，应该分两阶段：先跑小空间出 baseline，再针对 rescue 失败的 prompt 扩大搜索

这些教训进 [2026-04-19_replan](2026-04-19_replan_next_phase.zh.md)。
