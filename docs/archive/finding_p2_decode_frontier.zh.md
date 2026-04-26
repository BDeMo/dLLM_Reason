# Finding：P2 阶段 Decoding 前沿 —— SC 不够，需 T7 self-distill

> 语言：中文 | English: TODO

**日期**：2026-04-26

**前篇**：
- [`finding_t6_training_ceiling.zh.md`](finding_t6_training_ceiling.zh.md) —— T6 训练消融 + hardset
- [`definitions_hard_sets.zh.md`](definitions_hard_sets.zh.md) —— 术语定义
- 数据：`runs/validation/t6_decode_ablate/`、`t6_passN/`、`t6_hardset/`

---

## TL;DR

P2 阶段验证：**T6 trained 模型 + decoding 策略**到底能把 fail rescue 推到哪。

**新数据**（full-scope 1319 prompts × T ∈ {0.3, 0.7, 1.0} × N=8）：

- **pass@N (oracle) ceiling: ~66%**（Full-SFT step_336 @ T=1.0）
- **SC@N (deployable) ceiling: ~38%**（Full-SFT step_336 @ T=0.7）
- **gap = ~25-30%**（业界典型 10-15%，我们大得离谱）

**结论**：单纯 sampling + majority-vote 远远不够。差距太大，必须**把 capacity collapse 进 model**：T7 self-distill。

---

## 完整数据矩阵（full scope, N=8 samples）

| Ckpt | T | pass@1 | pass@4 | **pass@8** | **SC@8** | gap | ok pass@8 | ok SC@8 |
|---|---|---|---|---|---|---|---|---|
| Full-SFT step_336 | 0.3 | 35.0% | 44.1% | 50.2% | 34.1% | -16.1% | 96.5% | 93.1% |
| Full-SFT step_336 | 0.7 | 35.3% | 54.1% | 61.3% | **38.4%** ★ | -22.9% | 98.3% | 94.7% |
| Full-SFT step_336 | 1.0 | 29.3% | 54.7% | **65.9%** ☆ | 36.6% | -29.3% | 98.7% | 95.6% |
| Full-SFT step_84 | 0.3 | 29.3% | 39.9% | 43.5% | 31.1% | -12.4% | 96.7% | 92.9% |
| Full-SFT step_84 | 0.7 | 27.5% | 47.7% | 56.8% | 32.0% | -24.8% | 98.9% | 96.3% |
| Full-SFT step_84 | 1.0 | 28.4% | 48.0% | 59.8% | 31.1% | -28.7% | 99.0% | 96.1% |
| LoRA r=1 step_336 | 0.3 | 20.8% | 32.6% | 39.6% | 21.1% | -18.5% | 98.2% | 94.6% |
| LoRA r=1 step_336 | 0.7 | 23.3% | 42.0% | 50.2% | 24.2% | -26.0% | 99.2% | 95.5% |
| LoRA r=1 step_336 | 1.0 | 25.1% | 47.7% | 56.5% | 28.1% | -28.4% | 98.9% | 96.5% |

★ = 当前最强 deployable（SC@N pareto pick）
☆ = 当前最强 oracle（pass@N 上限）

---

## 7 个关键洞察

### 1. pass@N 在 full scope 的真实上限是 ~66%（不是 30+30 测的 70-77%）

之前 30+30 subset 数据偏乐观（高方差小样本）。**真实 ceiling 65-66%**。LoRA r=1 的 30+30 数据 76.7% 在 full scope 落到 56.5%，更说明小样本不可信。

### 2. SC@N 远远低于 pass@N（gap 25-30%）

pass@N 假设有个神告诉你 8 个 sample 哪个对（oracle）。SC@N 只能 majority vote。

**gap 大意味着**：模型在 fail prompts 上**没有收敛到正解**。8 个 sample 里偶尔产生 1-2 个对的（pass@N 看到），但**剩下 6-7 个错的有系统性偏好**（同一个错答案）。majority 投了错的。

### 3. SC@N 比 T=0 greedy 只好 ~10%

```
Full-SFT step_336 T=0 pass@1 (canonical):  28.1% fail rescue, 91.6% ok
Full-SFT step_336 T=0.7 SC@8:               38.4% fail rescue, 94.7% ok
                                             ──── +10.3% ────  +3.1%
```

**SC 单独能 deploy，但提升远低于 capacity ceiling 65%**。剩下 ~28% rescue 在 oracle 上能拿到，SC 拿不到。

### 4. ok 集 SC 也丢 3-5%

```
ok pass@8 = 98.7%
ok SC@8   = 95.6%   ← 3% 原本能对的，因为 majority 错答案被淹没
```

部署 SC 时这些 ok prompts 会翻车 —— SC 不是免费的（pass@N 假设了 oracle picker）。

### 5. T=1.0 最佳 pass@N，但 T=0.7 最佳 SC@N

```
fail pass@8:    0.3 → 0.7 → 1.0   单调升 (50.2 → 61.3 → 65.9)
fail SC@8:      0.3 → 0.7 → 1.0   非单调  (34.1 → 38.4 → 36.6)
```

**T 越高 → diversity 越大 → mode 越分散 → SC 越差**。pass 和 SC 的最优 T 不同。

### 6. LoRA pass@N 上限和 Full-SFT 接近，但 SC 差更多

| | pass@8 max | SC@8 max | gap |
|---|---|---|---|
| Full-SFT step_336 | 65.9% | 38.4% | 27.5% |
| LoRA r=1 step_336 | 56.5% | 28.1% | 28.4% |

LoRA capacity ceiling 略低（-9%），SC 落差略大。LoRA 更适合 sampling-heavy 部署，但 deployment 走 SC 时收益没有特别突出。

### 7. ok=100% 不可达，约束改成 ok ≥ 99%

之前我们说 "ok=100% 下 fail 最多多少"。实际上没 ckpt 触到 100%。原因：base 在 ok 集上 T=0 是 100%（by construction），但 T>0 + N=8 偶尔会 8 次都漏掉对答案。

```
现实 Pareto under ok ≥ 99%:
  Full-SFT step_84 @ T=1: pass@8 = 59.8%, ok = 99.0%
  LoRA r=1 step_336 @ T=0.7: pass@8 = 50.2%, ok = 99.2%
```

---

## 可视化：四条性能线

```
fail rescue (%)
     │
  70 │     ★ pass@N (oracle, 65.9%)            ← 模型能力上限
     │
  50 │
     │     ☆ pass@4 (54.7%)                    ← N=4 已大头
     │
  38 │   ◆ SC@N (deployable, 38.4%)            ← 当前 deploy 能拿到的
  35 │   ▲ pass@1 (T>0, 35.3%)                 ← 单 sample T>0
     │
  28 │   ● T=0 greedy (28.1%, T6 best ckpt)    ← 当前 prod 基线
     │
   0 │   ○ T=0 baseline (0%, untrained LLaDA)  ← scope_fail 定义
     └────────────────────────────────────────────
        sample budget →
```

**T7 目标**：把 ★ 的 65% capacity collapse 进 ● 的 deterministic mode。期望：T=0 pass@1 从 28% → 50-55%（接近 SC@N 但 deployable @ 1× sample budget）。

---

## 为什么不是 BoN/Verifier 优先

候选路线对比：

| 路 | 训练成本 | inference 成本 | 期望 fail rescue |
|---|---|---|---|
| 现状 (T6 + greedy) | — | 1× | 28% |
| **T7 self-distill** | 1-2 天（gen + SFT）| **1×**（deploy 时 deterministic）| **45-55%** |
| ORM + BoN | 1 天 | 8× sample + 1× verifier | 50-60% |
| PRM + BoN | 1 周 | 8× sample + N× verifier | 55-65% |
| RL on correctness | 2-3 天 | 1× | 40-55%（不稳）|

**T7 deploy cost 最低**（1× sample），train infra 最现成（`t7_gen_correct_samples.py` + `t6t7_train.py`）。即使 expected gain 略低于 PRM，**ROI 最高**。

T7 之后再叠 ORM/PRM 还能 stack 收益。

---

## T7 plan

### 数据

- **Scope**：gsm8k train（2000 条），筛 baseline LLaDA T=0 错的（约 60-70% = ~1300 条）
- **Sampling**：T ∈ {0.7, 1.0} × N=8 → 每 prompt 16 个 sample
- **Filter**：is_correct(extracted_answer, gt) 通过的 sample 进候选
- **Selection**：pick = "shortest"（最短 trace 防 verbosity 偏置）
- **Cover rate 估**：单 prompt N=16 × T=2，cover ≥1 的概率应该 ~50-60%（基于 fail set 现有 pass@8 ~65%）

### 训练

- **Warm-start**：T6 best (Full-SFT step_336)
- **Strategy**：FSDP 全参 SFT（同 T6 path）
- **Steps**：1500（2-3 epoch on ~700 collected samples）
- **LR**：2e-5（同 T6）

### Eval

- canonical T=0 pass@1
- decode_ablate full sweep
- SC@N 对比 T6 → 检查 collapse 是否成功

### Pipeline 脚本

`scripts/t7_pipeline.sh` 一键 gen + SFT + eval。8 GPU prompt-sharded gen → 自动聚合 jsonl → FSDP SFT → eval。

**预计耗时**：
- Gen（gsm8k train ~2000 prompts × 8 GPU sharded × 16 samples）= ~1.5-2 h
- SFT（1500 steps）= ~30 min
- Eval = ~30 min
- **总 ~2.5-3 h**

---

## 风险 / 已知坑

1. **Cover rate 不够高**：如果 baseline pass@N 在 train set 上只 50%，T7 训练数据只有 1000 条，可能不够把 capacity collapse 进 mode。Mitigation：用 T6 ckpt 当 base（pass@N 高），不用 vanilla LLaDA。
2. **Distribution shift**：T7 数据是 T>0 sampled output，包含模型自己的 phrasing。SFT 会让模型更倾向这种 phrasing → 可能在某些 prompt 风格上偏离 base 分布。
3. **ok 集回流**：T7 SFT 主要学 fail prompts 的修复，可能再次破坏 ok（catastrophic forgetting v2）。Mitigation：T7 数据混入 ok 集的 trace（确保 ok 不退）。
4. **生成时 model 选择**：用 base LLaDA 还是 T6？T6 capacity 高（pass@N 65%），但有 distribution shift；base capacity 低（pass@N ~50% 估）但 phrasing 标准。**当前 plan：用 T6**。

---

## 实测后要回填的数字

```
T7 candidates after gen:    ?
T7 SFT loss curve:          ?
T7 T=0 pass@1 fail rescue:  ?  (target: ≥ 45%)
T7 T=0 pass@1 ok retain:    ?  (target: ≥ 95%)
T7 pass@N ceiling change:   ?  (vs T6 65%)
T7 SC@N change:             ?  (vs T6 38%)
```

跑完 fill in，本 doc 转 final 状态。

---

## 决策树

```
T7 跑完后:

if T=0 pass@1 ≥ 45%:
    → SUCCESS, deploy T7
    → 是否还要 BoN？看 SC@N gap 是否仍 > 15%

if 35% < T=0 pass@1 < 45%:
    → 部分成功，验证 hypothesis 但需要补
    → next: ORM + BoN on T7

if T=0 pass@1 < 35%:
    → T7 没 collapse 上去
    → 调试：lr 太高/低？data 不够多样？
    → 可能要 PRM-RL 路线
```

---

## 文件

| 路径 | 说明 |
|---|---|
| `scripts/validate/t7_gen_correct_samples.py` | 重写为 in-process + batched + sharded |
| `scripts/t7_pipeline.sh` | 端到端：gen → aggregate → SFT → eval |
| `runs/validation/t7_gen_<ts>/per_prompt/*.json` | 每 prompt 全候选 + chosen |
| `runs/validation/t7_gen_<ts>/t7_sft.jsonl` | SFT 训练数据 |
| `runs/training/v161_t7_<ts>/hf/` | T7 训完的 HF ckpt |
| `runs/validation/t7_eval_<ts>/` | canonical eval（T7 vs T6 vs base）|
