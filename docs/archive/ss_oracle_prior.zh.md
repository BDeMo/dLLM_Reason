# SS Oracle Rate —— 先验估计

> 语言：中文  |  English: *(TODO — EN mirror)*

**日期**：2026-04-16
**目的**：在 SS（strategy_search, B1）真正跑完之前，基于现有 A 轴 / H3 / P6 crossref 的经验数据，给出 SS oracle rate 的**先验估计区间**，回答"SS 大概率能不能破 91.67%"。

**口径**：n=60 fail group。术语参见 [`ablation_index.zh.md`](ablation_index.zh.md) §Setting。

---

## TL;DR

| 量 | 值 |
|---|---|
| **下界**（A-union ⊂ SS）| **13/60 = 21.67%** |
| **中位估计**（= 全方法 union）| **55/60 = 91.67%** |
| **上界**（加 template_position 新维度 0-3 条）| **55–58/60 ≈ 91.67% – 96.67%** |
| **Ceiling floor**（期望仍成立）| `{4, 5, 14, 41, 42}` 不变 → **≤ 55/60** 不破，除非有意外 |
| **FAIL18 视角** | 下界 13/18 = 72.2%，上界 13–16/18 = 72–88.9% |

**主要 claim**：SS **很大概率** ≥ 91.67%（即全方法 union 的水平），且有望因 `template_position` 新维度再加 1–3 条。但 ceiling 5 **极不可能**被破（需要重审核心假设）。

---

## 1. SS 搜索空间相对历次 A 轴的覆盖

SS 的 5D pruned default space（384 configs/prompt × 1152 samples）：

| 维度 | SS 值集 | 此前最全的实验 | Sub / Super set? |
|---|---|---|---|
| `block_length` | {16, 32, 64} | A4: {8, 16, 32, 64, short_then_long} | 真子集（丢 bl8 和 非均匀；但 bl8 只贡献 idx=53 一条 rescue，被 A5/A6 也覆盖 → SS 无损）|
| `template_name` | {baseline, cot_plain, cot_step, answer_marker, step_by_step_prompt} | A5: {baseline, cot_plain, cot_step, answer} | 真超集（SS 多了 step_by_step_prompt）|
| `template_position` | **{prefix, suffix_scaffold, mid_anchor, none}** | **无**（diffusion-LM 独有新维度）| 纯新增 |
| `gen_length` | {128, 160, 192} | A6: {64, 96, 128, 160, 192, 256} | 真子集（砍了 g64/g96/g256 这三档；但 A6 实测只有 g160 是甜点，其他都 ≤ baseline → SS 无损）|
| `temperature` | {0.0, 0.3, 0.7} | H3: {0.3, 0.7, 1.0}（T=0 不等同）| 部分重叠（SS 有 T=0 determinism，少 T=1.0；但 H3 本身 T=1.0 ≈ T=0.7 收益递减）|
| `pass@N` | T>0 各 N=4，T=0 N=1 | H3: T>0 各 N=8 | 每 T sample 减半；但 SS 总 sample 更多（1152 vs 24）|

**结论**：SS 覆盖范围**在任何单一轴上都 ≥ 此前实验的"有效部分"**（丢掉的点在此前实验里都 ≤ baseline）。

---

## 2. 下界 —— 几乎肯定 ≥ A-union 13/60

### 理由（反证式）

假设 SS rescue < A-union。那存在某 prompt `p*` ∈ A-union 但 SS rescue 不到。

- `p*` 被 A4 救 → 存在 `bl ∈ {16, 32, 64}` 和 template `baseline` 配置让它对（A4 rescue 集中 idx=0 实测 bl8 没贡献，其他都在 bl ∈ {16, 32}）。SS 完全包含这组配置 → 应该 rescue 到。
- `p*` 被 A5 救 → 存在 template ∈ {cot_plain, cot_step, answer} + bl32/prefix/g128 让它对。SS 同样包含 → 应 rescue 到。
- `p*` 被 A6 救 → 存在 gen ∈ {128, 160, 192}（g160 是甜点）+ bl32/baseline/prefix 让它对。SS 包含 → 应 rescue 到。

唯一的 rescue 丢失风险是：SS 的 `num_steps = gen_length` coupling 与原 A 轴实验的 step 预算归一处理有微小差异。但 E1 已证 `num_steps` 独立加步数 0 rescue，coupling 实验上无损。

### 量化

| 参考 | Rescue | 是否 ⊆ SS 覆盖 |
|---|---|---|
| A4 | 5 `{0, 8, 13, 15, 28}` | ✓ |
| A5 | 8（含 {0, 8, 10, 13, 15, 28, 53, 55} 等）| ✓ |
| A6 | 12 `{0, 10, 13, 15, 19, 28, 35, 48, 51, 53, 55, 59}` | ✓ |
| A-union | 13 | ✓ |

→ **SS rescue ≥ 13/60**（21.67%），置信度接近 1。

---

## 3. 中位估计 —— 全方法 union = 55/60

### 为什么 SS 能 ≥ H3 rescue

H3 在 n=60 上取得 52/60 = 86.67%，通过 N=8 × T ∈ {0.3, 0.7, 1.0} × bl32/baseline/g128 = 24 samples/prompt 的 diversity sampling 实现。

SS 的 T>0 采样量：
- 3 bl × 16 (template × position) × 3 gen × 2 T(>0) = **288 T>0 configs/prompt**
- 每 config N=4 samples → **1152 T>0 samples/prompt**
- **是 H3 的 48 倍**

H3 做到 52/60 时距离 ceiling 5 只差 3 条（{19, 48, 51} = A6-only）。这 3 条是 write-space lever 能救、diversity lever 救不了的结构性 rescue。

SS 包含 A6（gen_length）全部甜点维度，因此可以**同时用 diversity 和 write-space 两个 lever**。所以：

- SS 在 H3-coverable 的 52 条上 ≥ H3（采样更多）
- SS 在 A6-only {19, 51} 上 ≥ A6（有 gen=160 的配置）
- SS 在 {48}（A5+A6 救、H3 stuck）上也有 A5+A6 配置

即 SS 应 rescue ≥ **H3 ∪ A6-only = 52 + 2 + 1 = 55/60**（= 全方法 union）。

### 反向缺口风险

唯一风险：H3 在 T=1.0 下独立 rescue 某条而 SS 没有 T=1.0。但 H3 的 per-temperature 分析显示 T=1.0 的独家贡献很小（T=0.7 已经包含绝大部分 rescue），所以可忽略。

→ **SS rescue ≈ 55/60**，置信度约 0.85。

---

## 4. 上界 —— template_position 独家 rescue 0-3 条

### 这是 paper 的核心 claim

`template_position ∈ {prefix, suffix_scaffold, mid_anchor, none}` 是 **diffusion LM 独有**的 inpainting-style scaffolding。AR LM 只能做 prefix。如果 write-space 是 rescue 的本质杠杆（A6 已证），那 canvas-position-aware scaffolding **有理由** rescue 到 prefix-only conditioning 救不了的 prompt。

### 量化估计

| 情况 | P(事件) | 新增 rescue |
|---|---|---|
| template_position 0 独家贡献 | 0.25 | 0 |
| 1–3 条独家 rescue | 0.55 | +1 to +3 |
| 4+ 条独家 rescue（"inpainting 是第 5 个杠杆"）| 0.20 | +4 to +7 |

**期望值** ≈ +1.8 条，带较大 variance。

### 但 ceiling 5 依然期望不破

`{4, 5, 14, 41, 42}` 经过 A4/A5/A6/H3 四重方法仍不可救。template_position 是第 5 个独立维度 —— 理论上有可能打破 ceiling，但先验低：

- Ceiling 5 在 H3 pass@8（接近采样上限）下也 stuck → 这些 prompt 的 output 分布在所有配置下都不包含正确答案
- template_position 是结构性旋钮（类似 A4/A5/A6），不改变 output 分布的**内在采样支持**
- 例外：若 mid_anchor 注入的 template 文本**直接包含**让模型能链到正确答案的 scaffold，可能破 1–2 条。但这需要人工设计特定 template 匹配 prompt，而 SS 的 5 个 template 都是通用 CoT 诱导，不太可能恰好命中

→ **ceiling 5 期望仍守住**；SS 破 ceiling 是"意外惊喜"（先验 < 0.15）。

---

## 5. 组合先验分布

| SS rescue (n=60) | 百分比 | 解释 | P |
|---|---|---|---|
| < 13 | < 21.7% | 实验 pipeline 出 bug（不可能）| < 0.01 |
| 13–51 | 21.7–85% | SS 丢了 H3 的 diversity 信号（不太可能，采样量 48×）| 0.05 |
| 52–54 | 86.7–90% | template_position 无效 + 某些 A-union 未全 rescue | 0.15 |
| **55–57** | **91.7–95%** | **= 全方法 union，template_position 加 0–2 条** | **0.55** |
| 58–60 | 96.7–100% | template_position 是强信号 + 破 ceiling | 0.20 |
| 60/60 | 100% | "神奇"结果（ceiling 全破）| < 0.05 |

**点估计**：SS rescue ≈ **55–56 / 60 ≈ 91.7–93.3%**

---

## 6. 对 paper claim 的含义

**High-confidence claims (P > 0.85)**：
1. SS oracle rate **达到或超过** full-method union 91.67%
2. A-union 的每条 rescue 都在 SS 覆盖内（无 regression）
3. Ceiling 5 `{4, 5, 14, 41, 42}` 在 SS 下仍守住

**Medium-confidence claims (0.5 < P < 0.85)**：
4. `template_position` 独家 rescue ≥ 1 条 → paper "canvas-position-aware scaffolding" 论点得实锤
5. SS 在 FAIL18 子集达到 14-16 / 18 = 78-89%

**Risk-of-surprise**:
6. 如果 SS 实际 < 52/60（低于 H3），说明采样 pipeline 有 bug 或 search space 定义错误 → 先停下来查
7. 如果 SS **破 ceiling**（>55/60），说明 template_position 或某个维度交互超预期 → 重写 "capacity ceiling" 这部分叙事

---

## 7. 跑完后的验证清单

`ss_analyze.py` 输出里**必查**这几条：

- [ ] `oracle_rate_fail`：应 ∈ [0.87, 0.97]
- [ ] `ss_vs_a_union_new`：应 ≥ 0（无 regression）
- [ ] `ceiling_broken_by_ss`：应为空或 ≤ 1（期望 ceiling 5 守住）
- [ ] `inpaint_novel_set`：应非空（证 template_position 新信号）
- [ ] `oracle_rate_ok`：应 ≥ 0.95（pipeline 健康 sanity）

如果任一条偏离，先别下结论，先回看 shard log / scope 文件 / SS 配置是否符合预期。

---

## 附：估计方法论

- 下界：`SS ⊇ A-union`（纯容器论证，无假设）
- 中位：`SS ≈ full-method union` 由"SS 采样 48× H3" + "SS 覆盖所有 A 轴甜点"双重论证
- 上界：template_position 的独立贡献为条件概率 prior（无直接数据，靠 diffusion LM 理论 + A6 的 write-space 证据外推）
- Ceiling floor：基于 n=60 实验数据收敛 + write-space/diversity 解耦解释

一旦 SS 落地，**实测数据 override 所有先验**。本文档仅用于：(i) 判断 SS 是否跑 ok；(ii) 给 paper methods 章节一个"为什么这么设计搜索空间"的 ex-ante 理由。
