# Finding: Static DAG Search 在 gsm8k 上挽救率为 0

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
