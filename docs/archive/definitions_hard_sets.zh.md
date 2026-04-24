# Hard-set 定义与关系

> 语言：中文 | English: [definitions_hard_sets.md](definitions_hard_sets.md) (TODO)

**日期**：2026-04-24

**目的**：项目里有多个"hard"概念（Ceiling-5 / FAIL18 / hardset / capacity-ceiling），出自不同实验、不同 scope。本文档是**单一权威**定义，以后任何地方引用这些词都指向这里。

---

## 术语速查表

| 术语 | 大小 | 定义维度 | 数据来源 | 首次出现 |
|---|---|---|---|---|
| `scope_fail_60` | 60 | 旧 base scope | H0 forensics (2026-04-15) | A 轴早期 |
| `scope_fail_331` | **331** | 当前 base scope | `regen_scope.py` v1.6.1 (2026-04-21) | T6 era |
| `FAIL18` | 18 | `scope_fail_60` 里的挑战样本 | 人工选定 | A 轴 |
| `Ceiling-5` | 5 | **A 轴**（inference-only）任何方法都救不回 | finding_a_axis_exploration | 2026-04-16 |
| `T6_hardset` | **166** | **T6 训练轴**（24 ckpt × T=0 pass@1）都救不回 | finding_t6_training_ceiling | 2026-04-24 |
| `true_ceiling` | **?（未测）** | **所有轴联合**都救不回（training + decoding + verifier） | — | 待测 |

---

## 各集合的**实际定义**（可复现）

### `scope_fail_331`（当前 canonical fail 集）

**定义**：gsm8k test split（1319 条）在 **baseline LLaDA-8B-Instruct** 下用 canonical config（T=0, bl=32, g=128, low_confidence remasking, greedy argmax）推理后答错的 prompts。

**生成**：`python scripts/validate/regen_scope.py --mirror default`

**位置**：`runs/validation/scope_fail_prompts.json`

**不变量**：在这个 canonical eval config 下 baseline 的 fail_correct ≡ 0（by construction）。改 eval config 就是改 scope 定义。

### `scope_fail_60`（历史 scope）

v1.5 时期的 60-prompt 版本。A 轴 / H 轴所有"救不救得动"实验都基于这个 scope。与 `scope_fail_331` 的关系：不严格子集 —— regen_scope 在 v1.6.1 用新 canonical 重做了一遍，idx 0-59 仍对应同样 prompts（有意保持）。

### `FAIL18`

**定义**：`scope_fail_60` 里的 18 条挑战样本，用于 FAIL18 rescue 比例 metric：

```python
FAIL18 = {0, 4, 5, 8, 10, 13, 14, 15, 19, 28, 35, 41, 42, 48, 51, 53, 55, 59}
```

**用处**：所有 eval 脚本输出里 `fail18_rescued` / `fail18_rescued_count` 报的就是这个集合的 rescue 情况。

### `Ceiling-5`

**定义**：A 轴探索（inference-time 所有方法的 union）都救不回的 5 条：

```python
Ceiling_5 = {4, 5, 14, 41, 42}   # ⊂ FAIL18 ⊂ scope_fail_60
```

**出处**：`finding_a_axis_exploration.zh.md`，A 轴全部方法（A1 edge rewire, A2 token revise, A3 span revise, A4 block layout, A5 prompt template, A6 gen length, H3 pass@N）的并集剩下这 5 条。

**性质**：纯**推理时**（不动模型参数）的 ceiling —— 表示 base LLaDA-8B 在任何 decoding 变换下都无法正确回答这 5 条。

### `T6_hardset`

**定义**：24 个 T6 SFT ckpt（4 full-SFT + 20 LoRA rank×epoch）在 canonical T=0 pass@1 下**没有任何一个能救回**的 fail 子集。

**脚本**：`scripts/t6_hardset.py`

**大小**：166/331（50.2%），见 `runs/validation/t6_hardset/hardset.md`。

**性质**：**训练时**（training-axis）的 ceiling —— 在当前 teacher-trace SFT 数据下，任何 epoch/rank 组合都无法让模型 greedy 解出这 166 条。

### `true_ceiling`（尚未定义）

**构想**：A 轴 + T6 训练 + decoding strategy（pass@N / SC / BoN）联合都救不回的最硬子集。

**测法**：`t6_decode_ablate.sh` 跑完 + verifier ablate 后，取 `T6_hardset ∩ (decode_ablate 救不回)` 作为 true_ceiling 候选。

**未测**。

---

## 关系图

```
──────────── A 轴（inference-time）───────────
scope_fail_60 (60) ⊃ FAIL18 (18) ⊃ Ceiling-5 (5)

─────────── T6 轴（training-time）────────────
scope_fail_331 (331) ⊃ T6_hardset (166)

────────── 交叉验证（2026-04-24 实测）──────────
Ceiling-5 ∩ T6_hardset = ∅   ← A 轴 ceiling 的 5 条全部被 T6 某个 ckpt 救回
                               （step_336 破 3/5, step_84/672 各破 2/5, union = 5/5）

T6_hardset ⊃ (Ceiling-5 的补)  ← 166 条全是 T6 训不回的，与 A 轴 ceiling 不重叠

含义：A 轴 ceiling 和 T6 ceiling 是**正交**两个轴上的上限，不可互相推导。
```

**关键推论**：两个 ceiling 不同源 → 两个轴联合（training + inference）天花板比任一轴单独高。这就是 decoding ablate 要测的。

---

## 为什么有这么多 hard 集合？

因为 **ceiling 是相对某个方法类 M 定义的**：

```
Ceiling(M) = { p ∈ scope_fail : ∀ m ∈ M, m(p) ≠ gt(p) }
```

- `Ceiling-5` = Ceiling(A 轴所有 inference 方法)
- `T6_hardset` = Ceiling(24 个 T6 ckpt × canonical greedy decoding)
- `true_ceiling`（未来）= Ceiling(training + inference + verifier + tool-use)

每新增一类方法 M'，ceiling 定义式里的 M 变大，ceiling 变小。终极 ceiling 是 capacity 意义上的 —— 模型 pretraining 时没见过、或者 reasoning 能力根本不够的 prompts。

---

## 使用建议

- 写 paper 时**不要混用** "hardset" / "ceiling" —— 指明是哪个方法类的 ceiling
- 比如："T6-hardset (166)" 而不是 "hardset"
- 比如："A-axis ceiling (Ceiling-5)" 而不是 "ceiling"
- 如果指代 "全方法都救不回"，用 `true_ceiling`（待定义）

---

## 相关文档

- `finding_a_axis_exploration.zh.md` — A 轴 + Ceiling-5 来源
- `finding_t6_training_ceiling.zh.md` — T6 轴 + T6_hardset 来源
- `hypotheses.zh.md` — 所有假设登记簿
- `exploration_axes.zh.md` — A/B 轴索引
- `runs/validation/t6_hardset/hardset.md` — T6_hardset 166 条 idx 的具体 prompt 列表
