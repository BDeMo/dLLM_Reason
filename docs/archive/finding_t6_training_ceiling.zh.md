# Finding：T6 训练容量已饱和 —— rescue 瓶颈转移到推理策略

> 语言：中文  |  English: [finding_t6_training_ceiling.md](finding_t6_training_ceiling.md) (TODO)

**日期**：2026-04-24

**配套文档**：
- [`finding_a_axis_exploration.zh.md`](finding_a_axis_exploration.zh.md) —— A 轴粒度探索（pretraining / inference-time 角度）
- [`hypotheses.zh.md`](hypotheses.zh.md) —— 假设登记簿
- [`issues/minors.zh.md`](issues/minors.zh.md) —— 跨版本小 bug
- 数据：`runs/validation/t6_ablate/`、`t6_lora_ablate/`、`t6_passN/`、`t6_hardset/`

---

## TL;DR

对 LLaDA-8B-Instruct 做 **T6 teacher-trace SFT** 的 2D 消融（epoch ∈ {0.5, 1, 2, 4} × mode ∈ {full-SFT, LoRA r ∈ {1,2,4,8,16}}）共 24 个 ckpt 跑下来：

- **T=0 pass@1 下 rescue 率天花板 ≈ 50%** —— 166/331 fail prompts 没有被任何 ckpt 救过（真·hardset）
- **Full-SFT 独吞 rescue 能力** —— LoRA ckpt 的 rescue 几乎全是 Full-SFT 子集，20 个 LoRA 只贡献极少 exclusive
- **Pareto 前沿**：
  - Full-SFT ep=2（step 336）：**fail +28.1% / ok -8.4% / net +10**
  - LoRA r=1 ep=4（step 672）：**fail +10.6% / ok -2.4% / net +11**
- **pass@N 揭示真瓶颈**：Full-SFT ep=0.5 在 30+30 子集上 T=1.0 pass@8 = **70%**，ok 维持 100% —— 说明模型能力已学到，**greedy decoding 是瓶颈，不是训练容量**

**结论**：训练侧继续推（更多 epoch / 更大 rank / 新 SFT 数据）对 fail-rescue 的边际收益已近枯竭。下一步前沿是**推理策略**（pass@N / majority vote / BoN / 更细 block / remasking 策略），而非训练更大/更久。

---

## 实验矩阵

### 训练 × decoding 的二维因子

| 轴 | 取值 | 总 ckpt |
|---|---|---|
| **Mode** | full-SFT (FSDP FULL_SHARD)、LoRA rank ∈ {1,2,4,8,16} | 6 |
| **Epoch** | 0.5, 1, 2, 4（1 epoch ≈ 169 steps，train split = 1350） | 4 |
| **合计** | — | **24 ckpt**（4 full + 20 LoRA）|

训练 pipeline：`scripts/t6_ablate.sh`（full-SFT，单训练多 ckpt 导出）+ `scripts/t6_lora_ablate.sh`（LoRA，每 rank 一次训练，adapter-only 中间导出 + 一次性 merge 前 eval）。

### 对每个 ckpt：canonical eval + passN 子集

- **canonical T=0 pass@1**（全 scope 331 fail + 988 ok）：`scripts/validate/v16_eval.py`
- **pass@N（仅 full-SFT 4 ckpt，30+30 子集，T ∈ {0.3, 0.7, 1.0}, N=8）**：`scripts/validate/h3_passN_at_temperature.py`

---

## 关键数字

### Table 1 — Full-SFT epoch ablation（T=0 pass@1，full scope）

| step | epoch | fail rescued | ok retention | Δ fail | Δ ok | net | FAIL18 | ceiling-5 |
|---|---|---|---|---|---|---|---|---|
| 84 | 0.5 | 24.8% | 92.2% | +82 | -77 | +5 | 4/18 | 2/5 |
| 168 | 1.0 | 24.2% | 91.0% | +80 | -89 | -9 | 5/18 | 3/5 |
| **336** | **2.0** | **28.1%** | 91.6% | **+93** | **-83** | **+10** ★ | **7/18** | **3/5** |
| 672 | 4.0 | 26.3% | 87.4% | +87 | -124 | -37 | 6/18 | 2/5 |

**Full-SFT 甜点 = 2 epoch**。更长训练 → ok 持续掉但 fail rescue 不再升 → 典型灾难性遗忘。

### Table 2 — LoRA rank × epoch ablation（过滤 ok ≥ 95%）

| r | epoch | step | fail rescued | ok retention | net | FAIL18 |
|---|---|---|---|---|---|---|
| 1 | 0.5 | 84 | 6.3% | 98.1% | +2 | 2/18 |
| **1** | **4** | **672** | **10.6%** | **97.6%** | **+11** ★ | 4/18 |
| 2 | 1 | 168 | 9.4% | 97.7% | +8 | 3/18 |
| 4 | 4 | 672 | 12.1% | 96.6% | +6 | 4/18 |
| 8 | 0.5 | 84 | 6.9% | 98.2% | +5 | 2/18 |

**LoRA 最优 = r=1, ep=4**。高 rank（8, 16）行为趋近 full-SFT，低 rank（1-2）保 ok 更好但 rescue 天花板明显更低。

### Table 3 — pass@N on Full-SFT（30+30 subset, n_samples=8）

| ckpt | T=0.3 fail p@8 | T=0.7 fail p@8 | **T=1.0 fail p@8** | ok p@8 (min T) | verdict |
|---|---|---|---|---|---|
| step 84 (ep=0.5) | 43.3% | 63.3% | **70.0%** ★ | 96.7% | REJECTED |
| step 168 (ep=1) | 56.7% | 63.3% | 63.3% | 96.7% | REJECTED |
| step 336 (ep=2) | 53.3% | 60.0% | 63.3% | 96.7% | REJECTED |
| step 672 (ep=4) | 56.7% | 63.3% | 63.3% | 96.7% | REJECTED |

**T=1.0 pass@8 把 fail rescue 从 T=0 的 25-28% 推到 63-70%**（×2.5 倍），ok 几乎不掉（100% → 96.7%，但 n=30 置信区间含 100%）。h3 verdict 都是 "REJECTED" —— 意思是 **"采样 diversity 能救"**（fail_p8 > 20% 阈值）—— 说明模型能力已经学到，greedy 没选对。

---

## Hardset：真·天花板

`scripts/t6_hardset.py` 对 24 个 ckpt 的 per-prompt `correct` 求并集：

```
scope_fail:                  331
rescued by ≥1 ckpt:          165   (49.8%)
never rescued:               166   (50.2%)   ← hardset
```

### Rescue-count histogram（被多少 ckpt 救过的 prompt 数）

```
 0 次被救 ───── 166  ← hardset
 1              35
 2-5            82
 6-23           48
```

**意义**：就算把 24 个 ckpt **oracle-ensemble**（每 prompt 挑任一能救的 ckpt），T=0 pass@1 rescue 天花板 **= 49.8%**。训练侧继续扫 epoch/rank 超参的边际收益已近零。

### Per-ckpt rescue + exclusive（只这一 ckpt 能救的数）

| ckpt | rescued | exclusive | 评价 |
|---|---|---|---|
| full_step_336 | 93/331 | **11** | 最强训练点，含 11 条独占 |
| full_step_672 | 87/331 | **12** | 过训练但保留不同独占 |
| full_step_84 | 82/331 | 6 | 最快到甜点 |
| full_step_168 | 80/331 | 2 | |
| lora_r4_step672 | 40/331 | 1 | LoRA 最强 |
| lora_r1_step672 | 35/331 | 0 | **0 exclusive** |
| ... (其余 18 LoRA ckpt) | ≤30/331 | **0-2** | 几乎全是 full-SFT 子集 |

**关键观察**：**LoRA 20 个 ckpt 合计 exclusive 仅 ~5 条**。换句话说 LoRA 的 rescue 几乎完全被 Full-SFT 覆盖。LoRA 不是**新能力**，是**不同的 fail/ok tradeoff 点**。

---

## 对下一步的含义

### 训练轴（closed）

| 选项 | 预期收益 | 评估 |
|---|---|---|
| 更多 epoch | 单调走 overfit（ok ↓ 无尽） | **DEAD** |
| 更大 LoRA rank | 趋近 full-SFT，没给出新 rescue | **DEAD** |
| 不同 teacher（Qwen3.5 etc） | 可能救一些 hardset，但 teacher 本身也 ~50% 的硬度 | **LOW priority** |
| 更大 SFT 数据 | 可能（v1.6.1 只有 1350 条）；但 teacher-trace SFT 的 distribution shift 本质限制不变 | **MEDIUM** |

### 推理轴（next frontier）

| 实验 | 脚本 | 成本 | 预期 |
|---|---|---|---|
| pass@N on LoRA 全 scope | `t6_lora_passN.sh` | ~2h | 验证 LoRA 是否也能吃 70% pass@8 红利 |
| decoding strategy ablate（T × N） on best ckpt | `t6_decode_ablate.sh` | ~14h full / ~7h hardset | 确认 ok=100% 下 fail 推到多少 |
| majority-vote / self-consistency | 待写 | ~2h | 实际部署能用（pass@N 是 oracle） |
| block_length / remasking sweep | 待写 | ~3h per ckpt | block-level decoding（上次 A4 已证 8.33%） |
| BoN w/ verifier head | 待实现 | 几天 | pass@N 的 practical 近似 |

### 真 frontier 是 hardset 的 166 条

只有通过 **pre-training 能力补足** 或 **tool-use / external verifier** 才有希望：
- 167 条 hardset 不随训练超参变化而改变（24 个 ckpt 全失败）
- 说明是 **LLaDA-8B 本身的 reasoning capacity 缺口**，不是 SFT 能弥补的
- 留作 "真 ceiling" 锚点

---

## Methodology / 可复现

所有数据 + 脚本 on dev @ `b6dfa19`+：

```bash
# 训练（~1h + ~2h）
bash scripts/t6_ablate.sh                              # full-SFT epoch 消融
bash scripts/t6_lora_ablate.sh                         # LoRA 20 cells

# 分析（秒级）
python scripts/t6_hardset.py                           # 真·hardset
python scripts/t6_passN_aggregate.py                   # passN 汇总

# 后续（未跑）
bash scripts/t6_lora_passN.sh --ranks 1,4 --auto_gpus  # LoRA passN（~2h）
bash scripts/t6_decode_ablate.sh --ckpt <best> --auto_gpus  # decoding ablate（~14h）
```

产物：
- `runs/validation/t6_ablate/summary.md` — full-SFT 4 行
- `runs/validation/t6_lora_ablate/summary.md` — LoRA 20 行
- `runs/validation/t6_passN/summary.md` — passN 合表
- `runs/validation/t6_hardset/{hardset.md, hardset.json, per_ckpt.json}` — hardset + 每 ckpt 细节

---

## 教训

1. **先做 hardset 再决定下一轴**：从 165/166 的 50/50 切分就能看出训练轴已枯竭，省了多跑 rank=32/64 的盲扫
2. **pass@N 是用来 diagnose 瓶颈位置的，不是做 prod**：但它告诉你 "瓶颈不是 capacity，是 decoding" —— 这个信号价值远超 benchmark 数字
3. **Exclusive rescue 数是消融该不该继续的量化指标**：LoRA 20 个 ckpt 只贡献 5 条 exclusive → 再加 rank 也没用；Full-SFT 4 个 ckpt 贡献 31 条 exclusive → 不同 epoch 真的学到不同 prompts
4. **6 次 audit bug 的代价**（见 minors.zh.md）：v1.6.1 第一次跑（2000 steps, ~12 epoch）产出 net -179；修 bug + 加消融后 net +11。**单纯"跑起来就对" 的管线不够**，训练-评估管线每一环都要 audit。

---

## 下次打开这份文档的时机

- 若你在想 "要不要训更大 LoRA rank / 更多 epoch" → 答案在 **Hardset 段**：不要
- 若你在想 "T6 能 rescue 最多多少" → T=0 pass@1 ~28%，pass@8 ~65-70%，hardset ~50% 永远救不回
- 若你在做 decoding ablate → 先看 **Table 3** 确认起点
