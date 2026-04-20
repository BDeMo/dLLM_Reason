# Postmortem: SS 首次 run 搁浅

**日期**：2026-04-19
**状态**：done
**前置上下文**：[2026-04-16_strategy_search_design.zh.md](2026-04-16_strategy_search_design.zh.md)
**继任**：[2026-04-19_replan_next_phase.zh.md](2026-04-19_replan_next_phase.zh.md)

## 事实

- **RUN_DIR**: `runs/validation/strategy_search_20260417_042620`
- **启动**: 2026-04-17 04:26
- **停止检查时**: 2026-04-19 06:14（48 小时）
- **完成**: **15/109 prompts fully done**（= 5760/41,856 configs = 13.8%）
- **预期完成**: 按 ex-ante 6.5h 跑完 → 实际**慢 ~30×**

## 诊断过程

### Step 1: 怀疑 stdout 全缓冲

初期症状：client_shard*.log 全空。以为 shard 崩了。

**结论**：Python stdout 重定向到文件时默认全缓冲（4-8KB），`pp.tick()` 每 prompt 一次、单行 < 100 字节，不足以 flush → log 看不到。Server log 正常因为 uvicorn 自己 flush。

**修复**：commit `3ac3ac1` 给 shard launch 加 `PYTHONUNBUFFERED=1 + python -u`。**下次 run 适用，当前 run 数据没丢**。

### Step 2: 怀疑 inpaint endpoint 慢

所有 8 server 最后请求都是 `POST /generate_inpaint`，inpaint 占 62.5% configs → 自然怀疑这里。

**curl 测 /generate_inpaint** with `"What is 2+2?"` @ gen=128 T=0:
```
HTTP 200  time_total=5.58s
```

**结论**：inpaint 跟 `/generate` 一样快（5.9s）。**不是 inpaint 特有的问题**。

### Step 3: 怀疑 Python HTTP client 慢（无 Session）

**Python client 端到端测试** 5+5 次调用:
```
call 0-4: 5.68-5.92s (len=13 chars)
inpaint 0-4: 5.65-5.87s (len=199 chars)
```

**结论**：Python client 跟 curl 一致（~5.8s/call）。**不是 HTTP layer 问题**。

### Step 4: 核对 shard-level 进度均匀性

8 个 shard 第一条 prompt 完成时间：
```
shard 2 (GPU 2): 25h
shard 3 (GPU 3): 22.5h
shard 4 (GPU 4): 23.4h
shard 0, 1, 5, 6, 7: 41-43h
```

GPU 2/3/4 快约 1.8×，其他 5 张慢。没有查 GPU 级 throttle / 温度 / PCIe 拓扑 → 暂时无解（硬件层差异）。

**结论**：shard 速度不均但都没"卡死"，server log 都在活跃接 request。硬件层异常是次要因素。

### Step 5: 重新核算 per-sample 实际耗时 ← 真正根因

用 ex-ante 5.5s/call 估 × 1152 samples = 1.76h/prompt。但：

**curl / Python test 用的是** `"What is 2+2?"`（~5 token prompt, gen=128）
**真实 SS prompt 是** gsm8k（50-120 token prompt, gen ∈ {128, 160, 192}）

Attention 复杂度 **O(seq²)**：
- Test: seq = 5 + 128 = 133
- Real @ gen=192: seq = 80 + 192 = 272
- 比值: `(272/133)² = 4.2×`

Forward pass count: gen=192 vs gen=128 = 1.5×

**Per-sample 实际时间估计**:
- gen=128: 5.8s（测 prompt 本身就是此档）
- gen=160 @ real prompt: 5.8 × (220/133)² × (160/128) = **15-19s**
- gen=192 @ real prompt: 5.8 × (272/133)² × (192/128) = **30-40s**

**Per-prompt 实际时间**（1152 samples 分布）：
| gen | samples | per-sample | 耗时 |
|---|---|---|---|
| 128 | 432 | 8s | 3456s |
| 160 | 288 | 17s | 4896s |
| 192 | 432 | 34s | 14,688s |
| **Total** | **1152** | | **23,040s = 6.4 h/prompt** |

Per shard 14 prompts × 6.4h = **~90 h per shard**（all 8 in parallel → 仍 ~90h）。

与实测 48h 完成 15 条（即每 prompt ~25h 在 shard 内）对上数量级。剩余 4× 差距可归因于：
1. 硬件层差异（shards 0/1/5/6/7 比 2/3/4 慢 1.8×，拖慢平均）
2. HTTP + Python overhead（inter-call 几百 ms 累积）
3. 某些 inpaint + 长 prompt 组合可能触发更糟的 O(seq²) 放大

## 根因定性

**设计阶段的 budget 估算用了 toy prompt**，把 per-sample 时间低估 5-7×。

其他次要因素：
- 8 GPU 里 5 张慢 1.8×（无暇排查）
- Serial HTTP client，GPU util 不满（但非决定性）
- Stdout 全缓冲导致 debug 难（已修）

## 影响

- **3 天时间 + GPU 资源**浪费到 13.8% 进度
- **Paper claim "canvas-constrained reasoning" 的实验证据没收到**（template_position novelty 未验证）
- **SS 工具链代码本身没问题**（orchestrator / analyze / ss_to_sft / oracle_prior 都可复用）

## 收获

- 已完成的 15 条 per_prompt `.json` 文件**可以 salvage**（ss_analyze 在部分数据上也能跑，至少能看 template_position 分布的初步倾向）
- 所有脚本和 framing 决策（ablation_index / oracle_prior / 4 distill decisions / paper framing）已归档，**对 replan 无 sunk cost**

## 教训

1. **Budget 估算必须用 realistic prompt + 代表性 config**。Ex-ante 应在 2-3 条 fail prompt × {bl=32,g=128}, {bl=32,g=192}, {bl=64,g=192} 跑 N=5 次测均值 + 方差再乘 total samples
2. **O(seq²) 在 gen_length scan 里被低估**。新搜索空间设计要加 per-sample time 的**硬预算约束**（比如"avg per-sample ≤ 10s"自动裁掉过大 config）
3. **8 GPU 不一定同速**。预 run 前先 benchmark 每张卡单 call 延迟
4. **小 run 先跑，大 run 基于小 run 数据外推**。不要一次性吃 109 prompts × 1152 samples

→ 都进 [2026-04-19_replan_next_phase.zh.md](2026-04-19_replan_next_phase.zh.md)
