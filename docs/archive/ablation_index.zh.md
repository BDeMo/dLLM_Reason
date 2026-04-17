# 消融实验总索引

> 语言：中文  |  English: *(TODO — EN mirror)*

**日期**：2026-04-16（首版）
**说明**：本文档是所有消融/验证实验的常驻索引。新实验结束后 **append 一行**即可。具体方法/结果细节见各 finding/closure doc。
**口径**：所有数字以 **n=60 fail + 49 ok**（`runs/validation/scope_*_prompts.json`）为准，P6 crossref 权威。

---

## TL;DR

| 指标 | 值 |
|---|---|
| Scope | 137 条 gsm8k fail prompt（LLaDA-8B-Instruct, T=0, bl=32, g=128） → n=60 subset 实测 |
| Positive levers（SUPPORTED）| 4 条：A4 (8.33%), A5 (13.33%), A6 (20%), H3 (86.67%) |
| Dead paths（DEAD / REJECTED）| 4 条：A1 edge DAG, A2/H1 token revise, A3 span revise, E1 num_steps decouple |
| 全方法 union | **55/60 = 91.67%**（n=60 全集） |
| True capacity ceiling | **5 条** `[4, 5, 14, 41, 42]`（n=60 和 FAIL18 子集口径一致） |
| H3 ⊆ A-union | 19.2%（H3 跟 A 轴近乎正交） |
| 独家 rescue | A6-only `{19, 51}`（H3 也救不了）；H3-only 42 条全在 FAIL18 外 |
| "反 ceiling 候选"（H3 stuck ∩ FAIL18）| `{4, 5, 14, 19, 41, 42, 48, 51}` —— 其中 19/48/51 是 "A 轴能救但 H3 stuck"，反证 write-space lever 比 diversity lever 本质 |

---

## A 轴 —— Inference-time 采样 / prompt 干预

| 代号 | 实验 | 干预粒度 | 问题 | Verdict | Rescue | 脚本 | 归档 doc |
|---|---|---|---|---|---|---|---|
| **A1** | DAG search | edge（token pair） | 改 DAG 边能救 fail? | **DEAD**（三证：greedy/NAS/E2E）| 0/1319+200+106 | (多个) | `finding_dag_search_zero_rescue.zh.md` |
| **A2** = H1 | Single-token revise hook | 1 token | 低 conf token 置回 mask 能救? | **REJECTED** | 0/137 | `h1_remask_rescue.py` | `finding_a_axis_exploration.zh.md` §H1 |
| **A3** | Span revise | 4-token 窗口 | 窗口平均 conf 低处置回 mask 能救? | **REJECTED** | 0/60 | `a3_span_revise.py` | `finding_a_axis_exploration.zh.md` §A3 |
| **A4** | Block-layout rerank | 整块（8-64 tok） | 换 layout ensemble 能救? | **SUPPORTED** | **5/60 (8.33%)** | `a4_block_rerank.py` | `finding_a_axis_exploration.zh.md` §A4 |
| **A5** | Prompt-template rerank | 整条 prompt | 换前缀 (CoT / answer marker) 能救? | **SUPPORTED** | **8/60 (13.33%)** | `a5_prompt_template.py` | `finding_a_axis_exploration.zh.md` §A5 |
| **A6** | Gen-length rerank | 总 gen_length | 换长度 (64-256 tok) 能救? | **SUPPORTED**（A 轴最强单旋钮） | **12/60 (20.00%)** | `a6_gen_length.py` | `finding_gen_length_sensitivity.zh.md` |
| **A4×A5 joint** | Layout × template 6-cell | 6 格 ensemble | Overlap 预测成立? | 实跑验证完美 (10=10) | 10/60 (16.67%) | `a4x5_joint.py` | `finding_a4x5_overlap.zh.md` |

**A 轴结论（粒度阶梯）**：信号从 **block 级**开始出现，到 prompt/gen_length 级更强。token/span/edge 级全 dead → **模型没有"可局部修复的排序错误"**，只有全局重新配置（layout/framing/budget）才有效。

---

## H 轴 —— 假设验证

| 代号 | 假设命题 | 阈值 | Verdict | 关键数字 | 脚本 | 归档 doc |
|---|---|---|---|---|---|---|
| **H0** | Scope generation | — | — | 137 fail + 49 ok buckets | `h0_forensics.py` | — |
| **H1** = A2 | Revise hook (single-token) 有信号 | rescue ≥ 5% | REJECTED | 0/137 | 同 A2 | 同 A2 |
| **H2** | `order_var / content_var < 0.3` | ratio < 0.3 | REJECTED | **ratio = 0.754**（其实是 A4 信号预兆） | `h2_order_vs_content.py` | `finding_a_axis_exploration.zh.md` §H2 |
| **H3** | Pass@N capacity ceiling | `fail_p@8 < 5% ∧ ok_p@8 > 90%` | **REJECTED**（能力远没到上限）| **52/60 (86.67%)**，cross-axis 最强杠杆 | `h3_passN_at_temperature.py` | `finding_a_axis_exploration.zh.md` §H3 + `closure_a_axis.zh.md` §4.6 |

---

## E 轴 —— Diffusion LM 独有维度

| 代号 | 实验 | 问题 | Verdict | 结果 | 归档 doc |
|---|---|---|---|---|---|
| **E1** | `num_steps` 解耦 | 独立加 step 不改 gen_length 能救? | **REJECTED** | 0% rescue → 工程上固定 `num_steps = gen_length` | `finding_e1_e5_rules_out_latent_reasoning.zh.md` |
| **E5** | Latent reasoning | Diffusion LM 有 latent computation? | **REJECTED** | 无；rescue 效应来自 **write-space** 而非 latent compute | `finding_e1_e5_rules_out_latent_reasoning.zh.md` + `discussion_latent_space_reasoning.zh.md` |

---

## B 轴 —— 下阶段主线 / 未做

| 代号 | 实验 | 状态 | 脚本 / doc |
|---|---|---|---|
| **B1** | **Per-prompt strategy search**（5D：`block_length × template_name × template_position × gen_length × temperature`）| **IN PROGRESS**（多 GPU orchestrator 已上 dev，待首次 run） | `scripts/validate/strategy_search.py` + `scripts/validate/run_ss_shards.sh` |
| B3 | Tool-augmented re-checking | 未做 | — |
| B4 | Verifier head | 未做 | — |

---

## P 轴 —— 离线交叉分析（post-hoc）

| 代号 | 分析 | 状态 | 脚本 / doc |
|---|---|---|---|
| P4 | CoT 砸 12 条的 pattern 分析 | N=60 下 7 个 feature 全不显著，等 N=137 再跑 | `finding_p4_p6_feature_analysis.zh.md` |
| P5 | H3 × A 轴 crossref（初版）| **BUGGY**（schema 错配，silently 输出 h3_rescue=0，废弃） | `scripts/validate/p5_h3_crossref.py` |
| **P6** | H3 × A 轴 crossref（修正）| **权威**：full_union=55/60 (91.67%)，ceiling=5 `[4,5,14,41,42]`，h3_only=42，a6_only=`[19,51]` | `scripts/validate/p6_h3_crossref.py` |

---

## 预留代号（未启用）

- **H4**：learned policy head > oracle heuristic？（SS distillation 完之后开）
- **E2/E3/E4**：diffusion-specific 其他维度预留
- **B2**：reserved

---

## 关键交集（跨实验不变量）

以下 invariants 在 n=60 下**任何实验都不破**：

- **FAIL18**：`{0, 4, 5, 8, 10, 13, 14, 15, 19, 28, 35, 41, 42, 48, 51, 53, 55, 59}` —— bl32/baseline/g128 baseline 全错的 18 条，**A4/A5/A6 共用**
- **Ceiling 5 条**：`{4, 5, 14, 41, 42}` —— 全方法都救不了，在 n=60 和 FAIL18 两个口径下**都是这 5 条**
- **A6-only rescue**：`{19, 51}` —— 只有 gen_length 旋钮能救，H3 也救不了
- **H3-stuck ∩ FAIL18**：`{4, 5, 14, 19, 41, 42, 48, 51}` —— 8 条；去掉 ceiling 剩 `{19, 48, 51}` = "A 轴救但 H3 stuck"，反证 write-space 核心论点

---

## Verdict 术语表

| 术语 | 含义 |
|---|---|
| **DEAD** | 多次独立实现 / 多 seed 下一致 0 rescue，roadmap 去掉这条路线 |
| **REJECTED** | 按预设阈值不成立（注：REJECTED 不一定是"无信号"，如 H3 REJECTED 反而是好消息 —— capacity ceiling 不成立意味着还能救）|
| **SUPPORTED** | 按预设阈值成立，rescue 率 ≥ 阈值 |
| **INCONCLUSIVE** | 样本不足，需要扩数据 |
| **IN PROGRESS** | 脚本已上 dev、未跑完 |

## 轴分类含义

| 轴 | 含义 | 干预时机 |
|---|---|---|
| A | inference-time sampler / prompt 干预（通用） | 采样循环内 |
| B | 训练侧 / 外挂模块（verifier, tool, learned policy） | 训练 or 采样前 |
| E | diffusion LM 独有的采样维度（write-space, steps, latent）| 采样循环 or 建模 |
| H | 需要先验证的命题式假设 | 任意 |
| P | post-hoc 离线交叉分析，不动模型 | runs 落地后 |

---

## 如何 append 新实验

1. 选 axis（A/B/E/H/P）+ 下一个空闲代号（检查"预留代号"段）
2. 在对应表追加一行：`| 代号 | 实验名 | 粒度/问题 | Verdict | 数字 | 脚本 | doc |`
3. 如果改变了 invariant（ceiling 缩小、独家 rescue 变化），同步更新本文档 "关键交集" 段
4. 新建或追加对应 `docs/archive/finding_*.zh.md`

---

## 相关入口

- [`hypotheses.zh.md`](hypotheses.zh.md) —— 假设登记簿 + verdict board（H 轴细节）
- [`exploration_axes.zh.md`](exploration_axes.zh.md) —— A/B 轴索引 + 状态标签（旧，**本文档是更新版总表**）
- [`closure_a_axis.zh.md`](closure_a_axis.zh.md) —— A 轴 closure 完整叙事
- [`finding_a_axis_exploration.zh.md`](finding_a_axis_exploration.zh.md) —— A 轴 finding（每实验 design/result/caveat 细节）
- [`empirical_rescue_per_prompt.zh.md`](empirical_rescue_per_prompt.zh.md) —— FAIL18 逐条 per-prompt rescue 交叉表
