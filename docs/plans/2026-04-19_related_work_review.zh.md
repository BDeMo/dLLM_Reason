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

**Scope notes (as stated by authors)**:
- Role classifier trained on GSM8K-specific data; authors note broader taxonomies would extend coverage
- Five-role taxonomy characterized as "coarse" in the paper; authors position it as an initial formulation
- Evaluation focused on LLaDA-8B; extension to other diffusion LMs is noted as future work
- A consistency checker variant explored in the paper; authors report it reduced accuracy (reported as 64% → 3%) and is not part of the final method

**Baselines they compare against**:
- DOS: attention-matrix-based ordering (prior work on structural unmasking)
- d1 (84.5%), JustGRPO (89.1%): RL-trained methods. Authors note LogicDiff is complementary to RL approaches — their sampler can apply on top of RL-trained models

**跟我们的关系 —— 不同 axis，可组合**：

| 维度 | LogicDiff | Inpainting (我们) |
|---|---|---|
| 干预对象 | Unmask 顺序 (temporal) | Canvas 位置 pre-commit (spatial) |
| 粒度 | Token 级，5 semantic role 分类 | Canvas 级结构 |
| Setup | 需预训 4.2M 小 classifier | Inference-time structural pre-commit |
| Prompt 设置 | 0-shot per paper | 0-shot（我们 setup）|

**Paper 可以利用**：
- LogicDiff 是最近最相关的 diffusion-LM reasoning 工作，必 cite + 讨论
- 两种 intervention 作用在**不同 axis**，原理上可叠加（测试是 good future work）
- 我们的方法不需 task-specific fine-tune，与 LogicDiff 形成 methodological 互补

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

**Setup**: Per-dataset LoRA fine-tuning of the planner on oracle orders.

**Benchmarks**: GSM8K, MATH, Sudoku 9×9, StrategyQA. Models: LLaDA-8B, Dream-7B.

**跟我们的关系**:
- 不同 axis（temporal ordering via learned supervised planner）
- 他们的 **0.845 Gt-Margin oracle** 是 diffusion LM temporal unmasking 的有参考价值的 upper bound
- Canvas-constraint 是 orthogonal axis，理论可与 temporal ordering 组合

**Paper 可以利用**：
- 引 0.845 oracle 作为 temporal-axis upper bound
- 我们的 inference-time structural intervention 和他们的 distilled planner 形成 methodological 对照（structural vs learned）

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

**Setup**: RL-trained policy.

**跟我们的关系**：
- 同 temporal-axis family（学 unmask decision）
- Canvas-constraint 作用在 orthogonal axis (spatial)，与 RL-learned temporal policy 互补

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
- AR LM 架构，与 diffusion LM 路径不同
- 他们的 "templates" 是高层 problem-solving strategy（500 条 curated library），作用在 prompting + navigator；我们的 `template_name` 是 5 种 CoT-style suffix，作用在 canvas 的不同 position
- 同属"结构化推理"大方向的不同实现

**Paper 可以利用**：
- 背景 citation，说明 template-guided reasoning 在 AR LM 的成功激发了我们在 diffusion 路径上的探索

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

**Setup**: Distillation-based training (teacher CoT → student signal-aware model).

**跟我们的关系**：
- AR LM 架构
- 他们的 7 semantic signals 与 LogicDiff 的 5 roles 相似（semantic-bucket 级结构）—— 两者共同支持"coarse semantic structure 对 reasoning 有用" 这一 observation，与我们的 granularity-ladder 发现一致

---

### 8. TemplateRL (2505.15692)

**URL**: https://arxiv.org/html/2505.15692

**Claim**: Templates as guidance signals during RL training，Qwen2.5-Math-7B-Base 上 AIME24 +99.4%。

**Method**:
- Templates = action sequence (e.g. "Divide and Conquer", "Self-Reflection")
- 用 Problem Condition Complexity (PCC) 做 template retrieval
- Rollout 用 template 引导，RL 学习 validated strategic patterns

**Setup**: Template-guided RL fine-tuning (training-time).

**Numbers**: 在 Qwen2.5-Math-7B-Base 对比 GRPO 基线：AIME24 33.3% vs 16.7%，AMC 77.5% vs 55.0%

**跟我们的关系**：
- AR LM 架构
- 概念上类似（templates as guidance），但 intervention 时机不同：他们 training-time（RL rollout guidance），我们 inference-time（structural pre-commit）

---

### 9. Can Structured Templates Facilitate LLMs (2508.19069)

**URL**: https://arxiv.org/html/2508.19069

**Claim**: XML `<chain>...</chain>` 抽象 solution templates；U-shape scaling law by difficulty。

**Finding**: "Excessive low-difficulty data impedes abstraction；high-difficulty data enhances reasoning."

**Method**: 3-stage: SFT with weighted chain loss / prompt-time injection via LoRA / curriculum FT via GRPO

**跟我们的关系**：
- AR LM 架构
- 他们关于 "difficulty scaling" 的结论（对 reasoning 难度分布的敏感性）与我们 FAIL18 / n=60 分层思路在 spirit 上一致 —— 可作为对"子集选择"重要性的外部支持

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

**Models**: GPT-4o / GPT-4 / GPT-4o-mini (AR LM).

**跟我们的关系**：
- AR LM + KG retrieval，与我们路径不同
- 共同支持"structure-in-reasoning helps" 的大方向
- Paper 可引作背景：prior work integrates structure at retrieval / prompting layers; we contribute structure at the generation-substrate layer in diffusion LMs

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

### 1. 核心 contributions（我们 paper 独立贡献的维度）

1. ⭐⭐⭐ **Canvas-spatial constraints**（inpainting-style scaffolding）—— 当前 diffusion LM reasoning 文献里尚未被独立研究的维度
2. ⭐⭐ **Systematic granularity sweep** —— 跨 token/edge/block/prompt/gen/position 多粒度对比，是此前 diffusion LM reasoning 工作未覆盖的 breadth
3. ⭐⭐ **Inference-time structural intervention** —— 与已有 learned/classifier-based / RL-trained 方法形成方法论互补
4. ⭐ **Write-space vs diversity 现象** (A6-only {19,51} + H3 stuck) —— 观察到的实证信号
5. ⭐ **Capacity ceiling 定量界定** (5 条 prompt in n=60)

### 2. Related work 章节骨架

```
§2 Related Work
  §2.1 Unmasking-order interventions in diffusion LMs
    A rich recent line of work addresses the temporal ordering of
    token unmasking in masked diffusion LMs:
      - LogicDiff introduces a semantic-role classifier
      - DAWN extracts dependency structure from attention for speedup
      - Where-to-Unmask learns a supervised planner from oracle orders
      - Learning Unmasking Policies trains an RL sampler
    Our work is complementary: canvas-spatial constraints operate on a
    different axis and can in principle be combined with any of these.
  
  §2.2 Template- and scaffold-guided reasoning
    Template-based reasoning has been explored extensively in AR LMs:
      - ReasonFlux (thought-template retrieval)
      - Reasoning Scaffolding (semantic-signal distillation)
      - TemplateRL (template-guided RL)
      - Can Structured Templates Facilitate LLMs (XML chain SFT)
    We study structure at the canvas level, which is a generation
    substrate unique to diffusion LMs.
  
  §2.3 Constrained decoding
    Grammar- / schema-constrained decoding (Outlines, Guidance, JSON
    mode) constrain the output token distribution. Our canvas
    constraints operate on the generation substrate itself, a distinct
    (orthogonal) axis.
  
  §2.4 Causal and structured reasoning in LLMs
    Integration of explicit structure into LLM reasoning has been
    studied primarily for AR LMs:
      - Kiciman et al. (LLMs for causal discovery)
      - Causal Graphs Meet Thoughts (KG-guided retrieval)
      - CausalGraph2LLM (LLM graph-encoding sensitivity)
      - Structured Thinking Matters (KG via tool-calling)
    We extend the "structure-in-reasoning" theme to the diffusion
    substrate.

  §2.5 Foundational diffusion LMs
    Our work builds on the masked diffusion LM formulation
    (MDLM, Sahoo et al., NeurIPS 2024) and LLaDA.
```

### 3. Critical comparisons to run (paper 实验要求)

1. **Reproduce LogicDiff on our eval setup**（若他们 code 公开）——双向 reproduce，以便 apples-to-apples 比较：在他们 setup 下跑我们的方法，在我们 setup 下跑他们的方法
2. **Orthogonality verification experiment** —— 理论上 LogicDiff 的 temporal role-ordering 与我们的 canvas spatial pre-commit 作用在不同 axis，**组合是否比各自单独更好**是一个合理的实证问题

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

**我们口径**：baseline (T=0, bl=32, g=128, 0-shot) → 1319 − 137 = 89.6% 正确。跟 DAWN 的 5-shot / RL-trained 方法处同一 magnitude 段。

**关于 baseline 的说明**：不同 paper 使用不同的 prompting setup（0-shot / few-shot / gen_length / num_steps / 解码超参数），因此同一 model 在同一 benchmark 上可报出差异较大的 baseline 数字。LogicDiff 报告的 22% 是他们特定 setup 下的 baseline；我们的 89.6% 是在不同 setup 下的 baseline。两者 + 3-7 方法不可直接一行表横比。

**公平对比的做法**（paper 会说清楚的）：要么在他们 setup 下 reproduce 我们的方法，要么在我们 setup 下 reproduce 他们的方法，两种 setup 下分别汇报相对 gain（Δ over own baseline）。

---

## Updates

- **2026-04-19**: 首版 15 篇精读完成
- **2026-04-19**: 确认 LogicDiff 是 direct competitor 但正交 axis

## Retrospective（pending）

待 paper 写作时回填：
- 哪些 citation 真进了 final draft
- 哪些方法实际 reproduce 了作为 baseline
- Orthogonality claim 是否实锤（组合实验结果）
