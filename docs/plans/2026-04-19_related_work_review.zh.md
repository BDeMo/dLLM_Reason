# Related Work Review —— Structured Reasoning in Diffusion LMs

> 语言：中文  |  English: *(TODO — EN mirror)*

**日期**：2026-04-19
**状态**：active（持续 append new work）
**前置上下文**：[PROJECT_OVERVIEW](PROJECT_OVERVIEW.zh.md) 里 paper framing 定位 canvas-constrained reasoning；需要建立 citations + 了解 competitor landscape
**继任**：—

---

## TL;DR

**15 篇精读完**。关键发现：

1. **LogicDiff (2603.26771) 是 direct competitor** —— 同 model (LLaDA-8B) 同 task (GSM8K)，+38.7pp。但他们改 **temporal**（unmask 顺序），我们改 **spatial**（canvas 位置）→ **两个正交轴**。
2. **DAWN / Where-to-Unmask / Learning Unmasking Policies** 都在"learned unmask order"这条线，**跟我们正交**。
3. **ReasonFlux / Reasoning Scaffolding / TemplateRL** 都是 AR LM 路线，跟我们**不同架构**。
4. **Causal graph in LLM** 这条线（Causal Graphs Meet Thoughts, CausalGraph2LLM, Kiciman, survey）主要在 AR LM 上做 KG retrieval / prompt engineering，**不在 generation internals**。
5. 我们的 paper 核心 novelty：**canvas-spatial constraint** + **systematic granularity sweep** + **training-free**。

---

## 分类

### Tier 1 —— **Direct competitors (same diffusion LM + reasoning task)**

| 论文 | Venue | ArXiv | 跟我们的关系 |
|---|---|---|---|
| LogicDiff | 2026 | 2603.26771 | **DIRECT COMPETITOR**：同 LLaDA-8B + GSM8K，temporal role-ordering，+38.7pp |
| DAWN | 2026 | 2602.06953 | **Compatible**：同 LLaDA，但目标是 speedup 不是 accuracy |
| Where-to-Unmask | 2026 | 2602.09501 | **Compatible**：oracle-supervised learned unmask planner |
| Learning Unmasking Policies | 2025 | 2512.09106 | **Compatible**：RL 学 unmask policy |

### Tier 2 —— **Foundational / infrastructure**

| 论文 | Venue | ArXiv | 跟我们的关系 |
|---|---|---|---|
| MDLM | NeurIPS 2024 | 2406.07524 | **Foundational**：masked diffusion LM 理论底座 |
| LLaDA 原论文 | 2024 | — | **我们用的 model** |

### Tier 3 —— **Related work (AR LM, structured reasoning)**

| 论文 | Venue | ArXiv | 跟我们的关系 |
|---|---|---|---|
| ReasonFlux | 2025 | 2502.06772 | AR LM thought templates，间接相关 |
| Reasoning Scaffolding | 2025 | 2509.23619 | AR LM 7-semantic-signal distillation |
| TemplateRL | 2025 | 2505.15692 | AR LM templates + GRPO |
| Can Structured Templates... | 2025 | 2508.19069 | AR LM XML chain templates |
| Verifying CoT via Computational Graph | 2025 | 2510.09312 | AR LM 白盒 verification |

### Tier 4 —— **Background (causal graph / LLM)**

| 论文 | Venue | ArXiv | 跟我们的关系 |
|---|---|---|---|
| Causal Graphs Meet Thoughts | 2025 | 2501.14892 | KG + RAG, AR LM |
| CausalGraph2LLM | NAACL 2025 | 2410.15939 | 评 LLM 处理 causal graph 能力 |
| Causal Reasoning + LLMs (Kiciman) | NeurIPS 2023 | 2305.00050 | LLM 能做 pairwise causal discovery |
| Survey on Causal Reasoning LLMs | 2025 | 2503.09326 | 综述 |
| Structured Thinking Matters | 2025 | 2505.18034 | KG-via-tool-calling, Corr2Cause |

---

## Tier 1 详细精读

### 1. LogicDiff (2603.26771) ⭐⭐⭐

**URL**: https://arxiv.org/abs/2603.26771

**Claim**: Inference-time logic-role-guided unmask scheduler. 在 LLaDA-8B-Instruct frozen 的情况下，GSM8K 22.0% → 60.7%，MATH-500 23.6% → 29.2%。

**Method**:
- 训一个 2-layer MLP classifier（4.2M params = LLaDA base 的 0.05%），input = LayerNorm 过的 hidden state, output = 5 类 (premise / connective / derived / conclusion / filler)
- Training data: 7,473 GSM8K solutions, 891,432 tokens, 30 min on 1×H100，98.4% val acc
- Class weight: connective 10× （类别极不平衡，derived 93.6%, conclusion 3.9%, connective 1.3%, premise 0.8%, filler 0.4%）
- Unmask priority: `0.7 · role_order/(R-1) + 0.3 · (1-conf)` —— weighted, 不是严格 priority
- Role order: premise=0, connective=1, derived=2, conclusion=3, filler=4

**Experimental setup**: 256 denoising steps, 256 max new tokens, frozen LLaDA, <6% speed overhead

**Stated limitations**:
- "Role head trained on GSM8K only"
- "Five roles are coarse"
- "Evaluated only on LLaDA-8B"
- "Consistency checker failed" —— 他们尝试加个 remasking consistency 机制，accuracy 反而 64% → 3%

**Baseline comparison they made**:
- DOS (prior attention-based): "attention matrices as statistical proxies" —— LogicDiff 用 explicit classifier 打赢
- d1 (84.5%, RL-trained): complementary to LogicDiff
- JustGRPO (89.1%, RL-trained)

**跟我们的关系 —— 正交的两个轴**：

| 轴 | LogicDiff | Inpainting (我们) |
|---|---|---|
| 改什么 | unmask 顺序 | canvas 哪些位置 pre-committed |
| 粒度 | token-级 semantic bucket (5 类) | canvas-级 spatial structure |
| 训练 | 要训 4.2M classifier | training-free |
| Axis | Temporal (顺序) | Spatial (空间) |

**Paper 可以利用**：
- LogicDiff 是我们最强的 comparator，必 cite + compare
- 声称两个 axis 正交（理论上可组合）
- 我们 training-free + 泛化好（不用为每个 task 重训 classifier）

**BibTeX (draft)**:
```
@article{logicdiff2026,
  title={LogicDiff: Logic-Guided Denoising Improves Reasoning in Masked Diffusion Language Models},
  author={...},
  journal={arXiv preprint arXiv:2603.26771},
  year={2026}
}
```

---

### 2. DAWN (2602.06953) ⭐⭐

**URL**: https://arxiv.org/abs/2602.06953

**Claim**: Training-free dependency-aware decoding, 1.80-8.06× speedup preserving quality.

**Method**:
- 从 inference-time attention weights 抽 dependency graph（最后几层 all heads 平均）
- Threshold τ_edge → 稀疏有向图；filter attention sinks（outlier detection）
- 三步 pipeline:
  1. Dependency graph construction (thresholded attention)
  2. Anchor-guided decoding (high-conf τ_high=0.9 + induced by already-unmasked)
  3. Conflict-based scheduling (greedy max independent set)

**Numbers**:
- LLaDA-8B, GSM8K (5-shot): 77.94% accuracy, 4.33× speedup
- LLaDA-1.5, MBPP: 37.60%, 8.06×
- Dream-v0-Base, HumanEval: 39.63%, 3.29×

**Training-free**, no model modification.

**Baselines compared**: Fast-dLLM, KLASS, LocalLeap. Does NOT mention LogicDiff / Where-to-Unmask.

**跟我们的关系**：
- 正交 axis：DAWN 追求 speedup，我们追求 accuracy
- DAWN 的 dependency graph 来自 attention，是 **inference-time 自动构造**（我们 A1 edge DAG search 是 external search）
- DAWN 的 attention-derived graph 不是给 reasoning 用的，是给并行化用的

**Paper 可以利用**：
- 背景 citation "prior work on dependency-aware decoding (DAWN) addresses speedup; we address accuracy via structural canvas constraints"
- DAWN 的 attention-sink filtering 可能对我们有用（如果 mid_anchor 落到 sink 附近）

---

### 3. Where-to-Unmask (2602.09501) ⭐⭐

**URL**: https://arxiv.org/abs/2602.09501

**Claim**: Ground-truth-guided oracle unmask order, distilled to learned planner.

**Method**:
- **Gt-Margin**: `s_i = μ_θ(x_0^i | x_t) - max_{v≠x_0^i} μ_θ(v | x_t)` —— correct token 的 prob 减掉 best alternative
- Oracle: training 时每步选 Gt-Margin 最高的位置 unmask
- 学一个 supervised planner (LoRA) 学 oracle 排序 via PiRank/NDCG@k

**Numbers (LLaDA-8B on GSM8K)**:
- Gt-Margin (oracle, 训不到): 0.845
- Learned planner: 0.705
- Heuristic Margin (baseline): 0.605

**Not training-free** —— 要 LoRA on 每个 dataset。

**Benchmarks**: GSM8K, MATH, Sudoku 9×9, StrategyQA. Models: LLaDA-8B + Dream-7B.

**跟我们的关系**:
- 正交 axis (temporal learned policy)
- **Oracle 0.845 是 diffusion LM 的 unmask-order 上限参考**
- 如果结合 canvas constraint + 他们的 oracle → 理论可能 > 0.845

**Paper 可以利用**：
- 0.845 oracle 作为"structure-only upper bound"
- 我们 training-free，他们需 LoRA

---

### 4. Learning Unmasking Policies (2512.09106) ⭐

**URL**: https://arxiv.org/abs/2512.09106

**Claim**: RL-trained unmask policy，single-layer transformer mapping conf → unmask decision.

**Method**:
- Masked diffusion sampling formalize 成 MDP，dLLM 是 environment
- Policy: single-layer transformer, input = token confidences, output = unmask decision
- 用 RL 优化 sampling procedure

**Numbers**: 具体 benchmark 数字 abstract 未给。
"Match SOTA heuristics when combined with semi-AR (block) generation, outperform in full-diffusion setting."

**Training required**, not training-free.

**跟我们的关系**：
- 同 temporal-axis 路线（学 unmask decision）
- 我们还是正交到 spatial axis

---

## Tier 2 详细精读

### 5. MDLM (2406.07524) ⭐ Foundational

**URL**: https://arxiv.org/abs/2406.07524
**Venue**: NeurIPS 2024

**Claim**: Simple masked diffusion LM 比想象的好。Rao-Blackwellized objective，简化为 classical masked LM loss mixture。

**为什么 foundational**: 
- LLaDA 基于 MDLM 思路扩出来的
- 他们的 sampler 支持 arbitrary length + semi-AR generation
- 我们的 "block_length" 参数来自这个设计

**Benchmark**: 语言建模 perplexity benchmarks, approach AR perplexity.

**Paper 可以利用**：
- 引用为 "our work builds on the masked diffusion LM formulation (MDLM, Sahoo et al. NeurIPS 2024)"

---

## Tier 3 详细精读

### 6. ReasonFlux (2502.06772)

**URL**: https://arxiv.org/html/2502.06772v1

**Claim**: 500 thought templates + hierarchical retrieval, 91.2% MATH / 56.7% AIME24.

**Method**:
- Template library (500 条) 从 MATH + 中国高中竞赛 curate
- 每个 template: name / tags / description / scope / steps / examples
- Navigator (Qwen 32B) 配 trajectory → retrieve templates → instantiate
- **AR LM** (Qwen2.5-32B-Instruct 家族)

**跟我们的关系**：
- AR LM → 不同架构
- "Templates" 类似我们的 `template_name` dimension，但他们是 500 条，我们 5 条
- 他们是 retrieval + AR instantiate，我们是 prompt-level ensemble

**Paper 可以利用**：
- 背景 citation "template-guided reasoning in AR LMs (ReasonFlux); we study canvas-level templates in diffusion LMs"

---

### 7. Reasoning Scaffolding (2509.23619)

**URL**: https://arxiv.org/html/2509.23619v1

**Claim**: Distill teacher CoT trace 到 7 discrete semantic signals。

**Method**:
- 7 signal categories: contrast/addition/examples/opinion/reasoning/conclusion/response
- Teacher: Deepseek-R1；Student: Qwen2.5 0.5B/7B/14B
- Dual-objective training: token loss + signal prediction loss

**Numbers (Qwen 14B)**:
- StrategyQA 0.858 (vs 0.760 baseline SFT)
- GSM8K 0.942, MATH-500 0.928

**AR LM，training-required**.

**跟我们的关系**：
- AR LM → 不同架构
- "7 semantic signals" 概念上类似 LogicDiff 的 "5 roles"，都是**coarse semantic bucket**
- 验证了"coarse structure is useful" 这个我们的 granularity ladder 论点

---

### 8. TemplateRL (2505.15692)

**URL**: https://arxiv.org/html/2505.15692

**Claim**: Templates as guidance signals during RL training，Qwen2.5-Math-7B-Base 上 AIME24 +99.4%。

**Method**:
- Templates = action sequence (e.g. "Divide and Conquer", "Self-Reflection")
- 用 Problem Condition Complexity (PCC) 做 template retrieval
- Rollout 用 template 引导，RL 学习 validated strategic patterns

**AR LM (Qwen/Llama), training-required**.

**Numbers**: Qwen2.5-Math-7B-Base vs GRPO: AIME24 +99.4% (33.3% vs 16.7%), AMC +40.9%

**跟我们的关系**：
- AR LM → 不同架构
- "Templates as guidance" 概念类似，但他们 RL-training-time，我们 inference-time

---

### 9. Can Structured Templates Facilitate LLMs (2508.19069)

**URL**: https://arxiv.org/html/2508.19069

**Claim**: XML `<chain>...</chain>` 抽象 solution templates；U-shape scaling law by difficulty。

**Finding**: "Excessive low-difficulty data impedes abstraction；high-difficulty data enhances reasoning."

**Method**: 3-stage: SFT with weighted chain loss / prompt-time injection via LoRA / curriculum FT via GRPO

**跟我们的关系**：
- AR LM，training-required
- "Difficulty scaling law" 对我们 **FAIL18 vs n=60** 的 stratification 思路有启发

---

### 10. Verifying CoT via Computational Graph (2510.09312)

**URL**: https://arxiv.org/html/2510.09312v1

**Claim**: 白盒分析，把 MLP 替换成 transcoder → attribution graph → 分类判断每步 correctness。

**Method**: CRV (Circuit-based Reasoning Verification). 4-stage: transcoder → attribution graph → structural fingerprint → gradient boosting classifier.

**Benchmarks**: Synthetic Boolean, Synthetic Arithmetic, GSM8K.

**Model**: Llama 3.1 8B Instruct, **AR**.

**跟我们的关系**：
- AR LM，不同轴（verification not intervention）
- "Computational graph" 跟 "causal graph" 名字像，实质是 model internal circuit
- 有趣 future work：把 CRV 思路应用到 diffusion LM 看 step-level correctness

---

## Tier 4 详细精读

### 11. Causal Graphs Meet Thoughts (2501.14892)

**URL**: https://arxiv.org/html/2501.14892v2

**Claim**: KG-based causal retrieval for LLM CoT, +7% MedMCQA on GPT-4o.

**Method**:
- Causal function 给 relation 打权重，过滤出 causal subgraph
- CoT 按 "→" 分段 → entity recognition → 连边 → path pooling + scoring
- Causal graph 引导 **retrieval**，不是 generation

**AR LM (GPT-4o / 4 / 4o-mini)**.

**跟我们的关系**：
- AR LM + KG retrieval，跟我们不同路径
- 但"causal structure improves reasoning"的 trend 支持我们 framing
- Paper 可引作背景："prior work integrates causal graphs at retrieval layer (CGMT); we integrate structural constraints at canvas layer"

---

### 12. CausalGraph2LLM (2410.15939) NAACL'25

**URL**: https://arxiv.org/html/2410.15939v1

**Claim**: 评估 LLM 处理 causal DAG 的能力，发现对 encoding 格式极敏感（60% deviation for GPT-4/Gemini）。

**Encodings tested**: JSON / adjacency list / adjacency matrix / GraphML / Graphviz DOT / single-node / multi-node descriptions.

**Finding**: Adjacency matrix 基本接近 random baseline (0.50)。GraphML 最好（Mistral 0.46 vs JSON 0.21）。

**跟我们的关系**：
- Evaluation paper，不是方法
- 提醒我们：如果 paper 里有"structure encoding"选择，要 ablate

---

### 13. Causal Reasoning + LLMs: Opening a New Frontier (2305.00050) Kiciman et al.

**URL**: https://arxiv.org/abs/2305.00050
**Venue**: NeurIPS 2023

**Claim**: Foundational paper on LLM causal reasoning ability. GPT-4 on pairwise causal discovery: 97% (+13pp)，counterfactual 92%，event causality 86%.

**跟我们的关系**：
- Background citation "LLMs capable of causal reasoning (Kiciman et al. 2023); we extend to diffusion LMs with structural constraints"
- 不直接竞争

---

### 14. Survey on Causal Reasoning LLMs (2503.09326)

**URL**: https://arxiv.org/html/2503.09326v1

**Taxonomy**:
- Domain-Knowledge-Driven: experts / contextual / pre-defined prompts / fine-tuning
- Model-Driven: causal graph construction / causal effect estimation / counterfactual reasoning

**跟我们的关系**：
- 不讨论 diffusion LMs（明确指出）
- 可引为"prior surveys cover AR-LM causal reasoning (Yao 2025); we contribute the diffusion-LM perspective"

---

### 15. Structured Thinking Matters (2505.18034)

**URL**: https://arxiv.org/html/2505.18034v1

**Claim**: Tool-calling 生成 KG (JSON) → 再用 KG 回答 causal query. Corr2Cause F1 32.71 → 48.26 (Qwen3-32B).

**跟我们的关系**：
- 两段式 pipeline，不同方向
- "Structured thinking" 这个概念跟我们的 framing 有共鸣，但实现完全不同

---

## Paper positioning implications

### 1. 核心 claim 排序（我们 paper 的 uniqueness）

1. ⭐⭐⭐ **Canvas-spatial constraint (inpainting as scaffold)** —— 所有 cited work 里**没有**
2. ⭐⭐ **Systematic granularity sweep**（token/edge/block/prompt/gen/role/position）—— LogicDiff 只 1 种
3. ⭐⭐ **Training-free** —— LogicDiff / Where-to-Unmask / Learning-Policy 都要训
4. ⭐ **Write-space > diversity** story (A6-only + H3 stuck)
5. ⭐ **Capacity ceiling** 定量分析 (5 条)

### 2. Related work 章节骨架

```
§2 Related Work
  §2.1 Unmasking-order interventions in diffusion LMs
    - LogicDiff (semantic role)
    - DAWN (dependency-aware, speedup-oriented)
    - Where-to-Unmask (ground-truth oracle + distill)
    - Learning Unmasking Policies (RL)
    → all temporal (order); we are orthogonal (spatial canvas)
  
  §2.2 Template / scaffold-guided reasoning in AR LMs
    - ReasonFlux (500 thought templates, retrieval)
    - Reasoning Scaffolding (7 semantic signals, distill)
    - TemplateRL (templates + GRPO)
    - Can Structured Templates Facilitate LLMs (XML chains)
    → all AR; canvas constraint unique to diffusion
  
  §2.3 Constrained decoding (output-space)
    - Outlines / Guidance / grammar / JSON mode
    → output-space constraints; we introduce substrate-space
  
  §2.4 Causal-graph-guided reasoning
    - Kiciman (LLMs for causal discovery)
    - Causal Graphs Meet Thoughts (KG + retrieval)
    - Survey / CausalGraph2LLM
    → all AR + prompt-level; we go to generation internals

  §2.5 Foundational diffusion LMs
    - MDLM (Sahoo et al. NeurIPS 2024)
    - LLaDA
```

### 3. Critical comparisons to run (paper 实验要求)

1. **LogicDiff as baseline** —— 他们 code 公开的话 reproduce 一下在我们的 eval setup（n=60 fail 口径）
2. **Orthogonality test** —— 理论上 LogicDiff + 我们 canvas 可以叠加，如果能跑出"组合优于各自单独" → 强 positive result

### 4. 数字对照表（paper results 可借鉴的）

| Method | Model | GSM8K | 备注 |
|---|---|---|---|
| LLaDA baseline (LogicDiff setup) | LLaDA-8B | 22.0% | 0-shot? 他们的 baseline |
| LogicDiff | LLaDA-8B | 60.7% | +38.7pp |
| DAWN | LLaDA-8B | 77.94% | 5-shot setup |
| Where-to-Unmask oracle | LLaDA-8B | 84.5% | oracle, upper bound |
| Where-to-Unmask learned | LLaDA-8B | 70.5% | LoRA trained |
| d1 (RL) | LLaDA-8B | 84.5% | RL |
| JustGRPO | LLaDA-8B | 89.1% | RL |

**我们口径**：baseline (T=0, bl=32, g=128) → 1319 - 137 = 89.6% 正确. 跟 DAWN / JustGRPO 同级，远高于 LogicDiff 的 baseline 22%。

**说明**：LogicDiff 和我们用的是不同 baseline setup（prompting / gen_length / num_steps 等都不同）。他们的 +38.7pp 是在他们特定弱 baseline 上的提升。我们直接对比要么 reproduce 他们 setup，要么在我们 setup 跑他们方法。

---

## Updates

- **2026-04-19**: 首版 15 篇精读完成
- **2026-04-19**: 确认 LogicDiff 是 direct competitor 但正交 axis

## Retrospective（pending）

待 paper 写作时回填：
- 哪些 citation 真进了 final draft
- 哪些方法实际 reproduce 了作为 baseline
- Orthogonality claim 是否实锤（组合实验结果）
