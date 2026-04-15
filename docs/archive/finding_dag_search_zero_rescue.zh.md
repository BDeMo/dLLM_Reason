# Finding: Static DAG Search 在 gsm8k 上挽救率为 0

> 语言：中文  |  English: [finding_dag_search_zero_rescue.md](finding_dag_search_zero_rescue.md)

**日期**：2026-04-15
**包版本**：`dllm-reason v1.5.3`（`pyproject.toml`）
**状态**：Archived — 作为 Phase 1 pivot 的核心依据

---

## 1. 实验环境（完全复现所需）

### 1.1 模型 & 推理配置（`configs/eval_default.yaml`）

| 项 | 值 |
|---|---|
| `model_id` | `checkpoints/llada-instruct` |
| `torch_dtype` | `bfloat16` |
| `num_steps` | 128 |
| `block_length` | 32（sequence 分 4 block，每 block 32 tokens） |
| `temperature` | **0.0**（greedy argmax） |
| `cfg_scale` | 0.0 |
| `remasking` | `low_confidence`（原生 llada 采样，无额外纠错） |
| `max_new_tokens` | 128 |

**序列布局**：`L = 128`（prompt 32 + gen 64 + padding 32，由 best_dag_edges/depth 推算确认）

### 1.2 搜索配置（`configs/search/greedy.yaml`）

| 项 | 配置值 | 实际运行值 |
|---|---|---|
| `method` | greedy | greedy ✅ |
| `budget` | 100 | **30**（CLI override） |
| `num_candidates` | 10 | 10 |
| `patience` | 5 | 5 |
| `initial_dag` | `cot`（CoT template warm-start） | — |
| `fitness` | accuracy | accuracy |
| `fitness_samples` | 50 | — |

### 1.3 搜索算法（`src/dllm_reason/search/greedy.py` — `GreedyEdgeSearch`）

**流程**：
1. **Template warm-start**：跑 `init_templates` 里所有模板（观察 `history[0].step=8`，说明评估了 8 个模板），取 fitness 最高的作为起点
2. **候选生成**（`_generate_candidates`）：每轮生成 `num_candidates=10` 个候选 DAG，每个候选是 *单条边的 add/remove*：
   - 60% 概率 add 一条随机边（若不成环）
   - 40% 概率 remove 一条随机现有边
3. **贪心评估**：遍历候选，**找到第一个 fitness > best** 就接受并重新生成候选（`break`）
4. **早停**：连续 `patience=5` 轮无改进退出

**关键性质**：
- **Fitness hill-climbing，无探索温度**
- **Early accept**：找到第一个改进就停，不看其它候选
- **Template warm-start 消耗 8/30 budget**（27%）
- **Single-edge neighborhood**：每步只能改 ±1 条边（相对 3072 条总边，扰动比 0.03%）

### 1.4 数据 artifact

| 路径 | 内容 |
|---|---|
| `runs/research_20260411_030422/stage2_discovery/search_histories/gsm8k/prompt_*.json` | 1319 条 prompt 搜索历史（每个含 `history` trajectory） |
| `runs/research_20260411_030422/stage2_discovery/episodes.db` | SQLite，1319 rows，每行含最终 DAG `dag_json` + output + correct |
| `runs/research_20260411_030422/stage2_discovery/best_dag_per_prompt.json` | prompt → best_strategy/correct/num_strategies_tried 索引 |

---

## 2. 核心事实

### 2.1 挽救率 = 0

（scope：`research_20260411_030422` 单次运行，1319 prompts）

|              | final_fail | final_ok | All  |
|--------------|-----------:|---------:|-----:|
| **init_fail** |      137   |    **0** |  137 |
| **init_ok**   |        0   |    1182  | 1182 |
| **All**       |      137   |    1182  | 1319 |

- 初始失败的 137 条，搜索 30 步后 **rescue = 0 / 137 = 0.0%**
- 初始成功的 1182 条，**0 条被搞坏**
- **搜索净 Δacc = 0.000 pp**

（更早的 1533/151 数字是多 run 聚合，单 run 数字以此为准）

### 2.2 Init_fail 的 fitness 恒定为 0

手抽两条（`prompt_0002`、`prompt_0007`）：

```
init:  fitness=0.00  edges=3072  step=8   (warm-start 选中 CoT 模板)
...
last:  fitness=0.00  edges=3073  step=30
fitness trajectory:  min=0.00  max=0.00  final=0.00
edges range:        3071 – 3073   (30 步内仅 ±1 条边扰动)
```

**所有 137 条 init_fail，23 步候选评估里 fitness 从未变过 0**。greedy 本身是 hill-climbing，在绝对平坦的 plateau 上无信号可走。patience 没用，因为没有"任何改进"触发过 reset。

### 2.3 搜索空间实际非常小

- 1319 条 prompt 的最终 best DAG 只有 **28 个 unique 结构**
- 其中 22 个只出现 1 次
- 主导模板 edges=3072, depth=4, max_width=32（即标准 4-block semi-AR），占 86%（1134/1319）
- 搜索基本 = "从 CoT 模板起步，扰动 ±1 条边找不到改进，返回原点"

### 2.4 "非默认 DAG acc=100%" 是 selection artifact

之前误读的"185 条非默认 DAG 100% acc"——实际是：
- Template warm-start 阶段就 hit 到一个非 CoT 模板 → init_fitness=1.0 → early-stop，保存那个模板
- 这些 prompt 对**所有 warm-start 候选模板**都能做对（属于 easy case），不是某个特殊 DAG "救" 了它们
- 相关性 ≠ 因果性

---

## 3. 结论

**在 v1.5.3 的 llada-instruct + T=0 + GreedyEdgeSearch(budget=30, single-edge mutation) 配置下，static position-level DAG search 对 gsm8k 最终 accuracy 没有因果影响**。

之前报告里 "+3.6pp" / "非默认 DAG 救 14%" 的说法**全部作废**，源自：
1. 把"greedy warm-start 首个 template 命中"误读为"search 收敛结果"
2. 混淆 `default_fp` 与 "semi-AR baseline" — 实际 default_fp 是 CoT template，不是 baseline

---

## 4. 为什么挽救率是 0？三个假设（未证伪）

### H1: Commit-once-never-revise 是 MDLM 采样瓶颈
`remasking="low_confidence"` 虽然允许 remask，但 llada 的 block-wise 策略一旦 commit 就不会跨 block 回改。错 token 固化后，DAG 只能影响"接下来哪个 mask 先填"，填的时候上下文已经污染，predictor 再怎么按 DAG 顺序来也吐同样的错。
→ **D / F / H 方向（correction head / PC corrector / CDD constraint）直接针对这个**。

### H2: T=0 + 双向 attention 让 unmask order 几乎失效
双向 transformer 在固定 "已 unmask 集合" 下产生固定 logits，argmax 不依赖顺序。DAG 能影响的只剩：
- 每步 unmask 的 batch 大小（即 level width）
- 哪个 position 属于哪个 level
而且 single-edge mutation 动不了 level 结构（3072 条边改 1 条，拓扑层几乎不变）。
→ 搜索步长 × 评估粒度 × 贪心接受三者叠加，信号被噪声淹没。

### H3: llada 在 gsm8k 的 137 条 init_fail 上达到能力上限
无论 order / remasking 怎么改，这些 prompt 需要的推理能力超出 llada-instruct 的 token-level 表征。
→ 只有训练端（reasoning reward RL / trajectory distill）能动。

---

## 5. 下一步探索（区分 H1/H2/H3）

| 实验 | 做法 | 证伪哪条 | 成本 |
|---|---|---|---|
| **D. Failing-case forensics** | 拿 10 条 `init_fail` 的 output + ground_truth 手 diff，看错在哪个 token、是早期/晚期错 | 直观判断 H1 vs H3 | 0.5 h（读 `episodes.db` 即可） |
| **A. Remasking ablation** | 给 137 条 init_fail 加一个最简 revise hook（conf < τ 时重采样整个 block），测 rescue | H1：rescue > 5% 证实 | 0.5 d |
| **B. Temperature sweep** | 对 137 条跑 T ∈ {0.0, 0.3, 0.7, 1.0}, N=8 次采样, 算 pass@N | H3：pass@N ≈ 0 → H3 成立；>0 → 采样 diversity 有价值 | 0.5 d（纯推理） |
| **C. Bigger DAG mutation** | 改 `_generate_candidates` 一次性动 10+ 条边 / 整个 level，重跑 greedy | H2：若仍 rescue=0 → order 表达力确实无用 | 0.5 d |

**建议顺序**：先 D（0.5h 零成本），再 A（证伪 H1 成本最低且信息量最大）。

---

## 6. 对 Plan 的影响

**降优先级**（若 D/A 证实 H1）：
- **G** Order-Token Joint Search
- **E** Prism Tree Search 的 DAG 变体
- `search/` 下的 differentiable / NAS / evolutionary — 都是 static DAG 空间上的搜索变体

**升优先级**：
- **D** BackPlay Correction Head — 正面攻击 commit-once 问题
- **F** PC Sampler + Duo Schedule — remasking 另一路径
- **H** CDD Constrained Sampling — content-adaptive state-level 约束

**保留但不以 DAG 为中心**：
- `search/` 保留为 template 生成工具（多 template warm-start 用）
- `scheduler/dag_scheduler.py` 保留作为 content-independent baseline

---

## 7. Artifacts

| 文件 | 内容 |
|---|---|
| `test.ipynb` | 初始发现（confusion matrix，part 1） |
| `test_dag_deepdive.ipynb` | 28 unique DAG 结构 + prompt 特征分析 |
| `test_dag_gain.ipynb` | 净增益再算（可复现） |
| `docs/archive/finding_dag_search_zero_rescue.md` | 本文档 |

**原始数据**：`runs/research_20260411_030422/stage2_discovery/`

**相关代码入口**：
- `src/dllm_reason/search/greedy.py::GreedyEdgeSearch`
- `src/dllm_reason/graph/templates.py`（warm-start 模板池）
- `configs/search/greedy.yaml`
- `configs/eval_default.yaml`

---

## 8. 复现证据：NAS Supernet 同样 0 rescue（2026-04-15）

**目的**：排除"greedy 陷入局部极值"这一可能，换一个完全不同的搜索算法（NAS supernet，带温度退火 + 梯度/熵驱动）再跑一次。

**数据**：`runs/research_20260415_035714/stage2_discovery/`
- 200 条 gsm8k prompt（不同子集）
- `search_method = nas`，`budget = 50`
- `metadata = {method: nas_supernet, num_spans: 8, span_size: 16}`
- 配置：T=0.0, block_length=32, num_steps=128（同原 v1.5.3 配置）

**结果**：

| 指标 | 值 |
|---|---|
| init baseline 正确 | 95 / 200 (47.5%) |
| init baseline 错误 | 105 / 200 |
| search 救回 (fail→ok) | **0 / 105** |
| search 打破 (ok→fail) | 0 / 95 |
| 净 Δacc | **+0.0 pp** |
| 所有 prompt 的 `best_dag_edges` | **全 = 0**（search 全部收敛到空 DAG）|
| history 中 fitness 有任何变化的 prompt | **0 / 200** |

**关键差异**：
- greedy (v1.5.3) 从 CoT 模板 3072 edges 出发 ±1 抖动 → 停在 3072 附近
- NAS supernet 从某处 supernet 出发温度退火 (τ: 1.24 → 0.1)，熵从 2.83 → 0.04 —— **熵降了，但 num_edges 全程 = 0**

**NAS history sample**（`prompt_0002`, init_fail）：
```json
[
  {"fitness": 0.0, "step": 0},
  {"fitness": 0.0, "step": 20, "h": 2.83, "tau": 1.24, "num_edges": 0},
  {"fitness": 0.0, "step": 40, "h": 1.63, "tau": 0.48, "num_edges": 0},
  {"fitness": 0.0, "step": 50, "h": 0.037, "tau": 0.10, "num_edges": 0}
]
```

**结论加强**：DAG 结构空间里"空 DAG"就是全局最优（在 T=0 llada-instruct + gsm8k 组合下）。不是 greedy 的问题，不是 budget 的问题，**是 DAG 轴整个就没有信号**。

这进一步排除了"搜索算法不够强"这一解释，H1 / H2 / H3 的证据地位不变，DAG 方向降权结论**加强**。

---

## 9. 第三次复现：E2E differentiable search 同样 0 rescue（2026-04-15）

**目的**：排除 "搜索 formulation 太离散"，换一个端到端可微的 formulation（Lagrangian dual + sparsity penalty + 温度退火）再跑一次。

**数据**：`runs/research_20260415_040451/stage2_discovery/`
- 106 条 gsm8k prompt
- `search_method = e2e`，`budget = 50`
- `metadata = {method: e2e, final_h: nan, final_tau: 0.10, total_steps: 50}`
- history 每步带 `lambda`, `rho`, `sparsity`, `num_edges`（典型可微搜索的 dual variable）
- 配置：T=0.0, block_length=32, num_steps=128（同 v1.5.3）

**结果**：

| 指标 | 值 |
|---|---|
| init baseline 正确 | 48 / 106 (45.3%) |
| init baseline 错误 | 58 / 106 |
| search 救回 (fail→ok) | **0 / 58** |
| search 打破 (ok→fail) | 0 / 48 |
| 净 Δacc | **+0.0 pp** |
| 所有 prompt 的 `best_dag_edges` | **全 = 0** |
| history 中 fitness 有任何变化的 prompt | **0 / 106** |
| num_edges 有任何变化的 prompt | **0 / 106** |

退火过程确实在跑（τ: 1.0 → 0.10），但 **num_edges 全程 = 0** —— dual optimizer 主动选择"空 DAG"为最优。

**三次 run 综合**（覆盖三种结构上完全不同的搜索算法）：

| Run | 算法 | N | rescue | 收敛 best_edges |
|---|---|---|---|---|
| `research_20260411_030422` | greedy ±1 edge | 1319 | 0/137 | ~3072（CoT 模板）|
| `research_20260415_035714` | NAS supernet | 200 | 0/105 | 0 |
| `research_20260415_040451` | E2E differentiable | 106 | 0/58 | 0 |

**结论**：跨三种结构上完全不同的搜索过程（组合 / supernet / 连续松弛），全部 **0 rescue**。其中两种主动选择空 DAG。这是 **H2**（T=0 + 双向 attention 让 unmask order 信号 ≈ 0）的独立强证据：在该配置下 `num_edges=0`（完全自由顺序）和 `num_edges=3072`（CoT 模板）等效。

DAG 方向降权结论现在**三重确认**，跨算法家族成立。Phase 1 在 static DAG 轴上的 search 变种（greedy / NAS / differentiable / evolutionary）都撞到同一个平坦 plateau，瓶颈必须在别处（sampler 或 训练）。
