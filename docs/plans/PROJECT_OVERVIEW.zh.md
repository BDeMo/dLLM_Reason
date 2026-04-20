# 项目总览 —— Purpose / Phases / Current Plan

> 语言：中文  |  English: *(TODO — EN mirror)*

**创建日期**：2026-04-19
**性质**：top-level 常驻文档（不带日期前缀）。新阶段开始时更新此文；详细 plan 仍放日期前缀的文件里。

---

## 1. 项目目的

**研究问题**：**离散扩散语言模型（discrete diffusion LM）** 在 reasoning 任务上答错时，inference-time 的什么**结构性干预**能救？

**目标 artifact**：
1. **一篇 paper**，定位 *canvas-constrained reasoning in discrete diffusion LMs*
2. **一个 policy head**（未来）：训练一个小模型学 `prompt → best strategy`，在推理时自动选策略

**实验床**：
- Model: `GSAI-ML/LLaDA-8B-Instruct`（当前唯一 open-source 8B discrete diffusion LM）
- Dataset: gsm8k（math word problems；baseline 错 137/1319）
- Scope: `scripts/validate/h0_forensics.py` 抽出的 137 fail / 1182 ok，实验常用 n=60 fail + 49 ok subset

---

## 2. 核心论点（paper claim）

### Framing: *"Canvas-Constrained Reasoning"*

**AR LM vs diffusion LM 的根本区别**：
- AR LM 的输出长度是 emergent 的（EOS 决定），canvas 是无限卷轴
- Diffusion LM 的输出长度是 **预先指定的硬约束**（`gen_length` 采样前给定），canvas 是**固定画布**

→ core constraint = **fixed finite write-space**（paper 名字的主卖点）

### 两个 core story

**Story 1 —— "Granularity ladder + structural vs dynamic constraints"**
- Token/edge/span 级**动态** revise 全 DEAD
- Block/template/gen_length 级**结构**干预都 SUPPORTED
- 结论：*"In discrete diffusion LM reasoning, only pre-committed structural constraints rescue errors; runtime confidence-based constraints carry no signal."*

**Story 2 —— "Canvas-position scaffolding"**（diffusion LM 独家）
- `template_position ∈ {prefix, suffix_scaffold, mid_anchor}` 是 inpainting-style scaffolding
- AR LM 只能 prefix；diffusion LM 能任意 position
- 如果这个维度有 **独家 rescue** → paper 核心差异化 claim

### 已知定量结果（n=60 fail）

| 量 | 值 |
|---|---|
| 全方法 union | **55/60 = 91.67%** |
| True capacity ceiling | **5 条** `{4, 5, 14, 41, 42}` |
| A-union (A4∪A5∪A6) | 13/60 = 21.67% |
| H3 pass@N (T ∈ {0.3, 0.7, 1.0}, N=8) | 52/60 = 86.67% |
| H3 ⊆ A-union | 19.2%（跨轴近乎正交） |
| A6-only rescue `{19, 51}` | 在 H3 下也 stuck → "write-space > diversity" 证据 |

---

## 3. 研究阶段（时间序）

### Phase 1 —— A 轴 discovery ✓ DONE (2026-04-15 ~ 04-16)

**目标**：扫 inference-time 干预的 granularity ladder

**做了**：A1 (edge DAG) / A2 = H1 (token revise) / A3 (span revise) / A4 (block layout) / A5 (prompt template) / A6 (gen length) / A4×A5 joint / H2 (variance) / H3 (pass@N) / E1 (num_steps) / E5 (latent reasoning)

**产出**：10+ findings，全方法 union 91.67%，ceiling 5 定量

**详见**：[`2026-04-15_a_axis_discovery_phase.zh.md`](2026-04-15_a_axis_discovery_phase.zh.md)

### Phase 2 —— H3 n=60 扩展 + P6 crossref ✓ DONE (2026-04-16)

**目标**：把初版 H3 n=30 补到 n=60 权威口径，写 crossref 工具

**产出**：H3 52/60 = 86.67%；P6 crossref 定 ceiling 5、A6-only、H3-only

### Phase 3 —— Strategy Search (per-prompt 5D) 🔴 **ACTIVE, 正在 replan**

**目标**：证明 **per-prompt 策略 > 任何 uniform 策略**；验证 `template_position` 这个 diffusion-LM 独有维度是否有**独家 rescue**

**首次尝试** (2026-04-17 ~ 04-19)：full 5D × 109 prompts × 1152 samples/prompt → **搁浅**（3 天只跑出 13.8%，root cause: ex-ante budget 用 toy prompt 低估 real gsm8k 的 O(seq²) attention cost 6×）

**Replan** (2026-04-19)：切到 **路线 B** —— FAIL18 × T=0 × gen ∈ {128, 160} × 5 template × 4 position × 3 bl = ~48 configs/prompt × 18 prompts，估 **4 小时跑完**

**详见**：[`2026-04-19_replan_next_phase.zh.md`](2026-04-19_replan_next_phase.zh.md)

### Phase 4 —— SFT Distillation ⏸ PENDING (等 Phase 3 数据)

**目标**：训一个小模型 learn `prompt → best_strategy`，跟 `uniform-best-strategy` baseline 比

**已决策**：cheapest winner / key=value 格式 / 裸 prompt input / `<UNSALVAGEABLE>` abstain（详见 [`2026-04-16_distillation_decisions.zh.md`](2026-04-16_distillation_decisions.zh.md)）

**工具已建**：`ss_to_sft.py`（`winners.json → sft_{train,val}.jsonl`）

### Phase 5 —— Paper 撰写 ⏸ PENDING

**framing** 已敲定 canvas-constrained reasoning（详见 [`2026-04-16_paper_framing.zh.md`](2026-04-16_paper_framing.zh.md)）

**骨架**：
- §1 Intro: canvas constraint motivation
- §2 Related: 跟 constrained-decoding (Outlines/Guidance) 的正交性
- §3 Methods: granularity ladder + 5D search space
- §4 Results: A 轴 + H3 + SS（Phase 3 数据）+ FAIL18 per-prompt
- §5 Discussion: structural vs dynamic constraints
- §6 Limitations: H4 learned policy / B4 verifier / B3 tool-use

---

## 4. 当前 (2026-04-19) 具体 action plan —— 路线 B

### Action 链（按执行顺序）

| # | 步骤 | 产出 | 时间 |
|---|---|---|---|
| 1 | 加 `--prompt_indices` CLI (支持 `fail18`/`ceiling5`/显式列表) | strategy_search.py 可精确切 FAIL18 | 10 min |
| 2 | 写 `probe_ss_benchmark.py` | 3 条真实 fail prompt × 代表 config，实测 per-sample 时间 | 30 min 写 + 10 min 跑 |
| 3 | Kill 旧 SS run + pull dev | 释放 8 GPU | 1 min |
| 4 | 跑 benchmark | 真实 per-sample 数字（avg + max） | 10 min |
| 5 | 决定是否砍 gen=192（如 benchmark 慢就砍）| final search space 敲定 | 5 min |
| 6 | 启 SS route B run | 18 prompts × ~48 configs × ~15s = 3-5h | 3-5 h |
| 7 | `ss_analyze.py` + `ss_to_sft.py` | analysis_report.md + sft_*.jsonl | 5 min |
| 8 | 定 template_position novelty 是否成立 | paper Story 2 成立/降级决策 | 人工 review |

**期望总耗时**：~5-7 小时（大部分是 Action 6 的等待）

### Decision fork at Action 8

- **If `inpaint_novel_set ≥ 2`** → Story 2 成立，paper 同时讲 Story 1 + Story 2（canvas-position scaffolding 是 diffusion-LM 独家）
- **If `inpaint_novel_set == 0`** → Story 2 降格，paper 只讲 Story 1（granularity ladder + structural vs dynamic）—— 仍是完整 paper，只是 claim 稍弱

---

## 5. 文档地图

### `docs/archive/` —— **研究事实**（永远 true）

- [`ablation_index.zh.md`](../archive/ablation_index.zh.md) —— **总索引**（所有实验代号 / verdict / 数字 / Setting & Definitions）⭐
- [`hypotheses.zh.md`](../archive/hypotheses.zh.md) —— H 轴假设登记簿 + verdict board
- [`closure_a_axis.zh.md`](../archive/closure_a_axis.zh.md) —— A 轴 closure 叙事
- [`finding_*.zh.md`](../archive/) —— 各实验的 design/result/caveat 细节
- [`empirical_rescue_per_prompt.zh.md`](../archive/empirical_rescue_per_prompt.zh.md) —— FAIL18 逐条 per-prompt 表
- [`ss_oracle_prior.zh.md`](../archive/ss_oracle_prior.zh.md) —— SS ex-ante oracle 先验

### `docs/plans/` —— **过程 / 决策 / 想法**（时间序）

- [`README.zh.md`](README.zh.md) —— 目录索引 + 维护规则
- **`PROJECT_OVERVIEW.zh.md`** —— 本文（top-level 常驻）
- `YYYY-MM-DD_*.zh.md` —— 分阶段/决策/postmortem

### 工具链

- `scripts/validate/strategy_search.py` —— 5D per-prompt 搜索
- `scripts/validate/run_ss_shards.sh` —— 多 GPU orchestrator
- `scripts/validate/ss_analyze.py` —— 跑完后的 paper-ready 7 段分析
- `scripts/validate/ss_to_sft.py` —— winners → SFT JSONL
- `scripts/validate/p6_h3_crossref.py` —— H3 × A 轴交叉分析（权威）
- `scripts/validate/aggregate_verdicts.py` —— hypotheses board 重写

---

## 6. 维护约定

- 每进入新阶段 → 更新本文 §3（Phase N 的状态）
- 每个具体 plan 有独立的日期文件 in `docs/plans/`
- 实验数据落地 → update `docs/archive/ablation_index.zh.md` 的对应表 + 涉及的 finding doc
- 本文 **不放数字以外的细节**（细节去具体 plan / finding）

---

## 7. 一句话摘要

> **我们证明了离散扩散 LM 的数学推理错误可以在 inference-time 通过 coarse-grained 结构干预救回 91.67%；硬天花板只有 5 条；token/edge 级 dynamic 干预全 DEAD；下一步是验证 diffusion-LM 独家的 canvas-position scaffolding 能否在 FAIL18 上再增一些独家 rescue，以锁定 paper 的核心差异化 claim。**
