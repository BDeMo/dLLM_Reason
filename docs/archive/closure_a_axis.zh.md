# A 轴阶段性收尾 —— 做了什么、为什么、接下来能做什么

> 语言：中文（主要） | EN mirror 按需生成

**日期**：2026-04-15（初版）/ 2026-04-16（H3 扩到 n=60 + P6 crossref 后同步）
**范围**：从 DAG search 证伪（v1.5.3）到 A3/A4/A5 + overlap 分析完成的一整条探索链。
**状态**：A 轴 inference-time 粒度扫描 **post-closure 全量结果出齐**。A6 gen_length 是最强 A 轴单旋钮（20%），**H3 扩到 n=60 后 pass@8 = 86.67%（52/60）成为最强单维度杠杆**，全方法 union 55/60=91.67%，true capacity ceiling 5/60 [4,5,14,41,42]。

> **2026-04-16 addendum**：初版（2026-04-15）基于 H3 早期 run (n=30)，数字 7/30=23.33%、full union 15/18=83.3%、ceiling [5,6,16]。H3 后续扩到 n=60 full fail set 重跑（`h3_passN_20260415_133254`），通过 `p6_h3_crossref.py` 得到以下权威数字。旧数字（n=30）在 Section 4.6 / 4.7 保留为 audit trail，新数字以 "n=60" 标注。

配套归档：
- [`finding_dag_search_zero_rescue.zh.md`](finding_dag_search_zero_rescue.zh.md) —— A1 三证死亡
- [`hypotheses.zh.md`](hypotheses.zh.md) —— 假设登记簿 + verdict 板
- [`exploration_axes.zh.md`](exploration_axes.zh.md) —— A/B 轴索引
- [`finding_a_axis_exploration.zh.md`](finding_a_axis_exploration.zh.md) —— A 轴详细设计和结果
- [`finding_a4x5_overlap.zh.md`](finding_a4x5_overlap.zh.md) —— A4 × A5 overlap 分析 + joint 6-cell 实跑验证
- [`retracted_broken_by_answer.zh.md`](retracted_broken_by_answer.zh.md) —— P2 诊断**撤回**（prompt 串线自证伪）
- [`finding_gen_length_sensitivity.zh.md`](finding_gen_length_sensitivity.zh.md) —— A6 gen_length rerank（SUPPORTED, N=60, rescue=20%）
- [`finding_p4_p6_feature_analysis.zh.md`](finding_p4_p6_feature_analysis.zh.md) —— P4/P6 离线特征分析（N=60 无显著信号，等 N=137）

---

## 1. 为什么要做这一轮（动机）

起点：v1.5.3 的 DAG-guided unmasking 在 gsm8k 上 **0 rescue**。greedy ±1 edge / NAS supernet / E2E differentiable **三个独立实现**都得出同样的结论：T=0 + 双向 attention 下，把 unmask 顺序建成一张图、在图上搜更优排列，对 correctness **无贡献**。

这把**核心研究假设**推翻了：DAG search 不是 paper 故事的 load-bearing 部分。剩下的选择有两个：

- **A**：在 inference-time 干预这条路上继续找 —— 如果 edge 级死了，换个粒度也许就活。
- **B**：放弃 inference-time 干预，转训练（SFT / RL）、工具、verifier。

B 轴代价大、反馈周期长。A 轴便宜、反馈快，**如果 A 轴也彻底死，才有理由转 B**。所以决策是：**先把 A 轴以粒度阶梯为轴，从细到粗系统扫一遍**。扫完得到的地图会告诉我们：
1. 如果全死 → 干净地 pivot B，附上 "inference-time 彻底没路" 的证据。
2. 如果有信号 → 知道信号在哪个粒度，下游知道怎么拼 sampler。

这是 **A 轴的全部目的**：用成本低的实验把"inference-time 还有没有活路"这个问题关死或打开。

---

## 2. A 轴的结构 —— 粒度阶梯

A 轴的设计是一根**从细到粗**的粒度梯度。每一级改变一个不同粒度的"干预单元"：

| 级别 | 实验 | 干预单元 | 假设形式 |
|---|---|---|---|
| edge | A1 | DAG 中的一条边（哪个 token 先 commit） | 改顺序能救 |
| token | A2 / H1 | 单个已 commit 的 token（conf 低就置回 mask） | 撤单 token 能救 |
| span | A3 | 4-token 窗口（窗口平均 conf 低就全置回） | 撤连续 span 能救 |
| block | A4 | 整个 block 的切分方式 | 换 layout 能救 |
| prompt | A5 | prompt 的输入 framing | 换模板能救 |

每一级的判据**同构**（rescue_rate ≥ 5% SUPPORTED / ≤ 1% REJECTED），用相同的 137-prompt fail 集（H0 产出的 `scope_fail_prompts.json`）做 one-to-one 比较。所以不同实验的结果可以**直接叠**（overlap 分析就是这种叠法）。

---

## 3. 做了什么 —— 每个实验一句话 + 结果

| 实验 | 做了什么 | N | 结果 | Verdict |
|---|---|---|---|---|
| **H0** | forensics：从 episodes.db 捞出 137 条 `correct=0` prompt，分桶存档 | — | `scope_fail_prompts.json` | DONE |
| **A1** (`finding_dag_search_zero_rescue`) | 三个独立 DAG search 实现：greedy ±1 edge / NAS supernet / E2E diff | 1319 + 200 + 106 | 三个都 0 rescue，NAS & E2E 自己选出 0 edges | **DEAD** |
| **A2 / H1** | 单 token revise hook：每 8 步，已 commit 且 conf<0.3 置回 mask | 137 | base=0, revise=0, rescued=0, broken=0 | **REJECTED** |
| **A3** | span-level revise：sliding window 窗口平均 conf<0.4 整窗置回 mask | 60 | base=42, revise=42, rescued=0, broken=0 | **REJECTED** |
| **A4** | block-layout rerank：5 种 layout (bl8/16/32/64/short_then_long) any-correct | 60 | base=42, any=47, rescued=5, rescue_rate **8.33%** | **SUPPORTED** |
| **A5** | prompt-template rerank：4 种前缀 (baseline/cot_plain/cot_step/answer) any-correct | 60 | base=42, any=50, rescued=8, rescue_rate **13.33%** | **SUPPORTED** |
| **A6** | gen-length rerank：6 种长度 (64/96/128/160/192/256) any-correct | 60 | base=42, any=54, rescued=12, rescue_rate **20.00%**；甜点 g160=49(81.7%) | **SUPPORTED** |
| **H2** | order_var vs content_var（扫 `block_length ∈ {16,32,64}` vs 扫 temperature） | 20 | ratio = 0.754（order 占 content 75%） | REJECTED (按阈值) |
| **H3** (n=30 初版) | pass@N at temperature，判 capacity ceiling | 30 (fail) | 7/30 rescue=23.33%；idx=0,2,8,13,15,24,28；4 stuck | **SUPPORTED** |
| **H3** (n=60 重跑) | 同上，扩到完整 60 条 fail | 60 (fail) | fail_p@8_max=86.67%；P6 rescue=52/60；8 stuck idx=[4,5,14,19,41,42,48,51] | **REJECTED**（按 hypotheses.md 的 capacity-ceiling 阈值 fail_p@8 > 20% → capacity ceiling REJECTED = 能力没到上限） |
| **A4×A5 overlap** | 两套 rescue 集交叉分析 | 60（60 共同） | independence=0.769；并集 10/18 = fail 的 55.56% | — |
| **A4×A5 joint** | 6-cell 实跑 `{baseline,answer}×{bl8,bl32,bl64}` | 60 | base=42, any=52, rescue=10, **完美验证 overlap 预测（10=10）** | — |

### A 轴一页总览

```
edge ────────── A1       DEAD     (0 rescue, 3 独立实现)
token ───────── A2/H1    REJECTED (0 rescue on 137)
span ────────── A3       REJECTED (0 rescue on 60)
block ───────── A4       SUPPORTED (5/60 = 8.33%)
prompt ──────── A5       SUPPORTED (8/60 = 13.33%)
gen_length ──── A6       SUPPORTED (12/60 = 20.00%) ← A 轴最强单旋钮
A4×A5 joint ── 6-cell   16.67% (10/60, overlap 预测完美验证)
pass@N ──────── H3       p@8=86.67% (52/60, n=60)  ← 最强单维度杠杆（正交于 A 轴）

rescue signal ↗ A 轴随粒度变粗而变强；A6 是 A 轴甜点
H3 (pass@N multi-temperature) 与 A 轴近乎正交（H3 ⊆ A-union 仅 19.2%）
全方法 union = 55/60 = 91.67%，true capacity ceiling = 5/60 [4,5,14,41,42]
```

---

## 4. 我们学到了什么 —— 核心结论

### 4.1 "confidence 当错误信号"这条路彻底死了

A2 (137 条, 0 rescue) + A3 (60 条, 0 rescue) 合起来：**无论 τ = 0.3 单点、还是 τ = 0.4 窗口平均，已 commit token 的 confidence 跟错误没有相关性**。模型在 fail 上是"confidently wrong"。

含义：**未来任何 sampler-side 纠错都不能用 committed-token conf 当信号**。候选替代信号：
- layout 间的 self-consistency（A4 里的 voting 思路）
- 独立 verifier head（B4）
- 工具重算（B3）

### 4.2 "inference-time 干预"的信号随粒度变粗而单调变强

```
edge → token → span → block → prompt → gen_length
 0      0       0    8.33%   13.33%    20.00%
```

这是 A 轴最干净的结论：**LLaDA-instruct 在 gsm8k 上没有可以被局部修复的排序错误**。没有 "某个 token 放错位置，搬回去就对了" 这类 local fix。有效的干预全部是**全局的**：要么重走整条 denoise trajectory（A4），要么重塑整个输入 framing（A5），要么给模型更合适的生成预算（A6）。

**A6 是最强单轴信号**：gen_length=160 单点（49/60=81.7%）已经超过 A5 的 any-template ensemble（50/60=83.3%）。g160 比 baseline g128 多了一个 block（5 blocks vs 4 blocks），刚好给模型足够的空间完成推理但不至于太长导致 trajectory 发散。

等价表述：**T=0 下 LLaDA 的行为更接近 block-wise quasi-autoregressive sampler**。块内顺序被 low-confidence-first 钉死（所以 edge 级没信号），块间顺序是 scheduler 选择（所以 block 级有信号），输入直接决定输出分布（所以 prompt 级信号最强），生成预算直接约束输出空间（所以 gen_length 信号最强）。

### 4.3 H2 在这套叙事里被 rehabilitate 了

H2 原来被按 `order_var / content_var < 0.3` 判 REJECTED（实际 0.754）。重新审视代码后发现：**H2 的 "order 轴" 其实就是 `block_length ∈ {16, 32, 64}`**，跟 A4 是同一个旋钮的子集。

所以 H2 的 0.754 **一开始就是 block-layout 级别的方差数字** —— 它预示了 A4 会有信号，只是当时按 "order vs edge" 的 framing 读错了。A4 的 8.33% rescue 是 H2 那 75% 输出方差里真正含 correctness 增益的部分。

A1 (edge 级 0) 和 H2 (block 级 0.754) 从来不矛盾 —— 它们在不同粒度上。

### 4.4 `answer` 模板是 trade 不是 free win —— 一个险些被写错的结论

A5 里 `answer` 模板是最强的单杠杆（`answer=45 > baseline=42`, net +3）。一开始写成"**one-template-for-all `answer` 可直接上线**"。但 overlap 分析（per-prompt 直接查）揭穿了：

```
baseline=T, answer=T:  37
baseline=T, answer=F:   5   ← 被 answer 砸
baseline=F, answer=T:   8   ← 被 answer 救
baseline=F, answer=F:  10
```

**`answer` 单模板是 5-for-8 交易**。在 fail-enriched 子集（60 条里 18 fail）上净 +3，在一般分布（base-correct 率 ~85%）上几乎肯定全局掉点。

**正确做法**：`{baseline, answer}` **2 格 ensemble**（让 baseline 接回被砸的 5 条），2× inference 拿到 A5 几乎全部 rescue 信号。

**方法论教训**：`broken=0` 是 ensemble-level 指标，不等于 per-template `broken=0`。per-prompt 必须直接查。

### 4.5 A4 和 A5 基本独立 —— 并不冗余

overlap 分析 `independence_factor = 0.769` —— 两轴在 rescue 上近乎不相交（1.0 = 完全不相交，0.5 = 完全重叠）。

- A4 独救 2 条（idx 13 → bl64, idx 53 → bl8）
- A5 独救 5 条（4 条里 `answer` 参与）
- 两者共救 3 条

**A4 ∪ A5 = 10 条 = fail 的 55.56%**。A4 alone 27.78% / A5 alone 44.44%，叠起来真有乘法增益。

能做的 ensemble 路线：
- **`{baseline, answer}` 2 格**：8/10，便宜可部署
- **`{baseline, answer} × {bl8, bl32, bl64}` 6 格**：10/10（理论），最可能落地的中档
- **5 × 4 = 20 格**：55.56% 上限，不可上线

### 4.6 H3 从初版 23.3% 扩到 n=60 后 pass@8 = 86.67% —— 最强单维度杠杆

**初版（n=30, 2026-04-15）**：H3 早期观察（17 条）预判 INCONCLUSIVE。30 条 fail 中 7 条 rescue（idx=0,2,8,13,15,24,28），rescue_rate=23.33% → SUPPORTED。4 条 stuck。

**n=60 重跑（2026-04-16, `h3_passN_20260415_133254`）**：完整 60 条 fail 集。
- `fail_pass@8_max = 86.67%`（T=1.0 下）/ `ok_pass@8_max = 100%`
- 按 hypotheses.md 的 capacity-ceiling 阈值（fail_p@8 > 20% → REJECTED）**H3 verdict = REJECTED**，即 capacity ceiling 假设不成立（能力没到上限）
- P6 crossref rescue 口径：H3 rescue = **52/60 (86.67%)**，H3 stuck = 8 条 idx=[4,5,14,19,41,42,48,51]
- 只 H3 救到的 (A4/A5/A6 都没救) = **42 条**；A6 独救 = [19, 51]；A4/A5 独救 = []
- H3 ⊆ A-union 仅 **19.2%**（52 条里只 10 条也被 A 轴救） → H3 与 A 轴近乎正交

**解读**：
1. capacity ceiling 假设被坚决 REJECTED —— 60 条中只 5 条是真 ceiling
2. H3 (multi-temperature + N=8 sampling) 是**最强单维度杠杆**，独立于 A 轴
3. 实操含义：如果愿意付 24× 推理代价（3 temps × 8 seeds），能救 86.67% 的 fail；A 轴只要 1× 就能拿 21.67%（A-union），但 A 轴上限止步于 13 条，要想再往上只能上多样性采样

**方法论修正**：初版"23.3% rescue"是 n=30 的偏小样本；新 n=60 数据大幅推翻了"pass@N 和 A 轴差不多"的直觉 —— 在完整 fail 集上，pass@N 实际上远超 A 轴全体。

**p5 bug 历史记录**：P5 (`p5_h3_crossref.py`) 写于 H3 旧 schema (pass_at_k dict) 时代，在当前 H3 的 `fail_XXXX.json + temps.T.pass@k` shape 下 silently 输出 h3_rescue=0。**P6 (`p6_h3_crossref.py`, 2026-04-16)** 是正确实现，输出 `runs/validation/p6_h3_crossref_h3_passN_20260415_133254.json`。

### 4.7 全方法 union 55/60 = 91.67%，只剩 5 条 true capacity ceiling（n=60 权威）

**初版（n=30）**：A4∪A5∪A6∪H3 = 15 ∪ FAIL18 外的 {2, 24}，分母是 18 (bl32/baseline/g128 都错的子集)，给出 **15/18 = 83.3%**。注意：分子包含 2 条 FAIL18 外的 H3 独救，分子分母口径不一致，是个毛刺。True ceiling 报为 3 条 [5, 6, 16]。

**n=60 权威（2026-04-16, P6 crossref 输出）**：

n=60 全集口径：
- A4: {13,15,28,53,59} (5)
- A5: {8,10,15,28,35,48,55,59} (8)
- A6: {0,10,13,15,19,28,35,48,51,53,55,59} (12)
- Joint 6-cell: {8,10,13,15,28,35,48,53,55,59} (10)
- **H3: 52 条 (86.67%)** —— 列表太长见 `p6_h3_crossref_*.json`
- A4∪A5 = 10, A4∪A5∪A6 = 13, **全 union = 55/60 = 91.67%**
- **True capacity ceiling = 5/60 = 8.33%，idx=[4, 5, 14, 41, 42]**

FAIL18 子集口径（= bl32/baseline/g128 三轴 baseline 一致错的 18 条：{0,4,5,8,10,13,14,15,19,28,35,41,42,48,51,53,55,59}）：
- A-union ∩ 18 = 13/18 = 72.2%
- H3 rescue ∩ 18 = 10/18 = 55.6% —— **10 条全被 A-union 涵盖**，H3 在 FAIL18 内没有独救
- **Full union ∩ 18 = 13/18 = 72.2%**（= A-union，H3 没加新的）
- Ceiling ∩ 18 = 5/18，idx=[4, 5, 14, 41, 42]

**两个 ceiling 数字一致**：true ceiling 5 条 {4,5,14,41,42} 完整落在 FAIL18 里。

**正交性**：H3 独救 **42 条全部在 FAIL18 外**（即 baseline 在三轴都对，但在 N=8 sampling 的 T>0 下偶尔会错，也偶尔会对），A6 独救 {19, 51} 在 FAIL18 内。H3 在 FAIL18 子集内 ⊊ A-union，但在 n=60 全集上和 A 轴近乎正交（pairwise 交集小）。

**旧 "H3∩A6 独有 {0}" 的新状态**：在 n=60 下 idx=0 ∈ A6 rescue ∩ H3 rescue，两轴同时救；不再是"H3∩A6 独有"。

### 4.8 A6 是 A 轴内最强单旋钮（n=60 H3 之后降为"A 轴内"最强）

gen_length sweep 结果：g64=27, g96=36, g128=42, g160=**49**, g192=39, g256=40。
- g160 是甜点：49/60=81.7% vs baseline g128=42/60=70%，单点 +11.7pp
- g160 单点已超过 A5 的 any-template ensemble（50/60）
- 不是"越长越好"：g192/g256 回落，说明有最优长度
- 注意：cross-axis 层面，H3 (pass@8 multi-T) 在 n=60 上达到 52/60 (86.67%)，远超 A6 全 sweep 的 12/60 (20%)；A6 最强仅限 A 轴内（单 T=0 单 seed 旋钮）

### 4.9 Joint 6-cell 完美验证 overlap 预测

A4×A5 的 `{baseline, answer} × {bl8, bl32, bl64}` 6-cell 实跑：rescue=10，跟 overlap 预测**完全一致**（10=10，零意外）。per-cell: bl8_baseline=43, bl8_answer=41, bl32_baseline=42, bl32_answer=45, bl64_baseline=37, bl64_answer=40。

验证了 overlap 分析方法论可靠、配置间无意外交互。

---

## 5. 没做成的事 / 已知局限

### 5.1 N 仍 = 60 不是 137，但信号已非常强

A3/A4/A5/A6 都在 N=60 上跑。但 post-closure 后信号远超阈值：A6 rescue=20% 是阈值 5% 的 4 倍，A5=13.33% 是阈值 2.7 倍，即使 N=137 上略有稀释也不会翻转 verdict。**N=60 上信号已经足够可靠，不再是 blocking issue**。

### 5.2 overlap 的 independence factor 是小样本估计

60 条里只有 18 条 fail。0.769 的 CI 很宽。但 joint 6-cell 实跑完美验证了预测（10=10），说明至少在这个 N 上 independence 估计是准确的。

### 5.3 H1 的 trigger 计数没持久化

`runs/validation/h1_remask_20260415_051706/summary.json` 里没有 "hook 触发了多少次" 的字段。之前口头说的 "122/137 没触发过" 无法从 summary 回溯验证 —— 只能从 correctness 向量推断 "hook 没发挥作用"，但"是 low trigger rate 还是 trigger 后 resample 收敛回"分不开。实际结论不变（REJECTED），但诊断精度打折。

### 5.4 A5 的 broken 模板没做深度诊断

- `cot_plain` −7、`cot_step` −12，`answer` 砸 5 条 baseline-对的 —— 这些 broken prompt 我们只有 idx + gt。没有深入看这些 prompt 的 baseline 输出 vs broken 输出、或跟 prompt 长度/算子数这类特征做关联。
- `cot_step` 尤其值得看：砸 12 条不是小数，里面可能藏着 "instruct 模型的答案格式预期"这类可学到的 pattern。

### 5.5 没尝试的粒度

在 "prompt" 这一级，我们只试了 4 种朴素后缀。没试过：
- Few-shot（在 prompt 里塞解答范例）
- 不同 system prompt
- 不同问题复述方式（paraphrase）

这些都是更重的 prompt-level 干预，理论上可能带来更大 rescue。但工程代价也大，而且偏离 gsm8k 的标准 eval 设定。

### 5.6 没做 A4 跟 A5 的"交互" ensemble

理论 20 格 ensemble 是纸上上限 55.56%。实际跑 20 次 decode、用某种 voting/majority 策略会是多少？我们没做。只做了 upper bound 的 overlap 计算（any-cell-correct）。

### 5.7 A4.1 预测器完全没做

A4.1 的想法是：train 一个 `prompt → best_layout` 的预测器，避免 5× inference 代价。但 overlap 分析后 A4 独立信号只有 2 条（N=60），N=137 放大到预期 5-8 条 —— 训不了分类器，只能试手工特征。**决策是降优先级**，但没完全放弃。

---

## 6. 接下来的主线 —— per-prompt strategy search pipeline

A 轴探索阶段已充分完结。全方法 union 15/18=83.3% 证明 inference-time 旋钮空间足够大，不需要 pivot B 轴。下一阶段从"哪个轴有信号"转到"对每条 prompt 找最优配置"。

### 主线：Per-prompt strategy search

对每条 prompt 搜最优 `(block_length × template × gen_length × temperature)` 组合：
1. **Strategy search**：遍历旋钮组合空间，存 `(prompt, best_strategy)` pairs
2. **新维度 template_position**：diffusion LM 特有的 scaffold/inpainting —— 把 template token 放在生成区域**任意位置**（不只是 suffix）。这是 AR LM 做不到的独特优势。
3. **训模型学策略**：用 search 产出的 pairs 训一个 `prompt → best_strategy` 预测器

### 已完结的旧优先级

| 旧编号 | 状态 | 说明 |
|---|---|---|
| P0 (N=137) | 降优先级 | N=60 信号已足够强（A6=20%, A5=13.33%），不再是 blocking issue |
| P1 (answer ensemble eval) | 被 strategy search 吸收 | 成为 strategy search 的一个特例 |
| P2 (broken-by-answer) | **DONE** | 见 `retracted_broken_by_answer.zh.md` |
| P3 (6-cell 实跑) | **DONE** | rescue=10，完美验证 overlap 预测 |
| P4 (cot_step pattern) | 等 N=137 | 7 个 feature 全不显著（见 `finding_p4_p6_feature_analysis.zh.md`） |
| P5 (H3 crossref) | **DONE** | H3 rescue=7（注意 p5 脚本有 BUG，手动算的） |
| P6 (A4.1 手工特征) | 等 N=137 | n=2 太小无意义 |
| P7 (gen_length) | **DONE → A6** | SUPPORTED, rescue=20%，最强单轴信号 |

---

## 7. 一句话总结

**A 轴 post-closure 全量结果：A6 gen_length 是最强单轴信号（20% rescue, g160=81.7%），H3 最终 SUPPORTED（23.3%），全方法 union 15/18=83.3% 只剩 3 条 true ceiling。不同维度基本正交，inference-time 旋钮空间巨大。下一步是 per-prompt strategy search：搜最优 (block_length x template x gen_length x temperature) 组合 + 训模型学策略。**
