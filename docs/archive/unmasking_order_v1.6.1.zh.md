# Unmasking Order in v1.6.1 Pipeline

**日期**：2026-04-20
**相关 plan**：`docs/plans/2026-04-19_v1.6_plan.zh.md`

本文**明确定义** v1.6.1 pipeline 的每一个环节里 unmasking order 是怎么定的，训练和推理时行为是否一致，以及跟 LogicDiff / DAWN 等现有工作的对比。

---

## 1. 一句话总结

- **训练（T6 SFT）**：unmasking order **隐式**、**均匀随机**（MDLM 标准）
- **推理（v16_eval.py / serve.py）**：unmasking order **显式**、**按置信度贪心**（`remasking="low_confidence"`）
- **训练数据里的 canvas 结构**（`<SETUP>/<STEP_N>/<ANSWER>`）**不直接**约束 order，但会**间接**影响推理时 order（置信度分布通过 SFT 被调整）

---

## 2. 训练时（T6 SFT loss）

标准 **masked diffusion LM** 目标。给定一条训练样本 `x_0 = concat(prompt, teacher_trace)`：

```python
t ~ Uniform(ε, 1-ε)                    # 随机时间步
mask_rate = t                           # 每个位置以概率 t 被 mask
x_t = where(rand() < t, MASK, x_0)     # 随机 mask (absorbing state)
logits = model(x_t)                     # forward
loss   = -log p(x_0 | x_t)              # 只在 (masked ∧ answer region) 位置
```

关键点：
- **mask 完全随机**，和 token 语义无关
- 模型被逼在**任意 random subset of positions**下填剩余
- **没有** "先 unmask setup 再 step 再 answer" 的显式训练信号 —— 所有位置平等

**为什么这还有用**：模型学到 `p(x_0 | x_t)` 的**联合分布**。如果 `<SETUP>...<SETUP>` 这些 structural tokens 在数据里高频 + 位置稳定（都在 canvas 前部），它们在任何 mask 下条件概率都高 → 推理时自然早被 commit。

---

## 3. 推理时（v16_eval.py / serve.py → LLaDAWrapper.generate）

**Canonical config**（v1.6.1 eval + scope 都用这个）：

```python
model.generate(
    prompt,
    generation_len=128,
    block_length=32,
    num_steps=128,
    temperature=0,
    remasking="low_confidence",
)
```

### Block-wise denoising

- `gen_length=128 / block_length=32 = 4 blocks`
- 每 block 独立 denoise，顺序**固定从左到右**（block 0 → 1 → 2 → 3）
- 一个 block 内部 `128 / 4 = 32 steps`，每 step commit 1 个 token

### Block 内 unmask order：low-confidence scheduler

```
for step in 1..32:
    logits = model(x_t)                 # forward on all positions
    conf[i] = max_softmax(logits[i])    # per-position confidence
    for each committed token: mask_prob_next = 0
    # 选 confidence 最高的 1 个 masked 位置 commit
    pick_i = argmax conf[i] over (i ∈ current_block ∧ i is masked)
    x[pick_i] = argmax logits[pick_i]   # greedy commit (T=0)
```

即 "low_confidence remasking" = **每步让 confidence 最高的位置落下来**（"low_confidence" 这个 flag 名容易误导 —— 实际是"把低 conf 的留着、高 conf 的先 commit"）。

### 跟训练的一致性

训练时位置是**random-mask**，推理时是**greedy-by-confidence**。训练分布**不完全对齐**推理，但这是 LLaDA 原生设计，不是我们 SFT 引入的偏差。

---

## 4. Canvas 结构跟 order 的间接关系

**命题**：SFT 在带 `<SETUP>...</SETUP><STEP_N>...</STEP_N>...<ANSWER>N</ANSWER>` 的 target 上训练，推理时 LLaDA 更容易按 **setup → steps → answer** 顺序 commit tokens。

**机制**：
- 训练数据里 `<SETUP>` 几乎总在 canvas 前部，`<ANSWER>` 在后部
- 训练后 `p(x_0 | x_t=ALL_MASK)` 对 `<SETUP>` 开头的位置给很高 confidence
- 推理第 1 步 → argmax confidence → commit `<SETUP>` 的第一个 token
- 后续 token 在已 commit 的 `<SETUP>` 条件下继续高 confidence，形成 "setup 段先落"
- 当 setup 段 commit 完，下一个 high-confidence 候选是 `<STEP_1>` 起始 token
- 最后 `<ANSWER>` 最晚落

**注意**：这**不是** LogicDiff 那种**显式 role-ordering 调度器**。我们没改 scheduler，只通过 training data 改 confidence 分布，**间接**引导 order。

**不保证**：
- 坏 prompt 下可能 `<ANSWER>` 反而 early commit（confidence 陡然高，模型瞎猜），然后 `<STEP_N>` 被这个假 answer 条件住 → 错
- 我们 v1.6.1 **没加机制阻止**这种 "answer-first collapse"

---

## 5. 跟 related work 的对照

| 方法 | Unmask order 来源 | 训练/推理一致？ |
|---|---|---|
| **LogicDiff** | Token-level role classifier (premise/connective/derived/conclusion/filler) → weighted priority = 0.7·role + 0.3·(1-conf) | 推理定制；训练只训 classifier，主 LM frozen |
| **DAWN** | 从 inference-time attention 抽依赖图 → 贪心独立集 | 都用同一套（training-free） |
| **Where-to-Unmask** | Oracle (Gt-Margin) 训一个 supervised planner | 训专门 planner |
| **我们 T6 (v1.6.1)** | **low_confidence 默认 scheduler（不变）** + 数据里 canvas structure 间接引导 | Training mask random ≠ inference greedy（标准 MDLM 假设不变）|

**我们不是**：
- ❌ 显式改 scheduler（ordering 逻辑跟 baseline LLaDA 相同）
- ❌ 训 planner / role classifier
- ❌ 强制 section-level 顺序

**我们是**：
- ✅ 改训练数据的**结构**，让 model 学到 canvas-structured distribution
- ✅ 依赖 LLaDA 原生 low-confidence scheduler 在推理时"自动" follow 结构
- ✅ Order 是 emergent，不是 engineered

---

## 6. 诊断 / 验证

想验证 "T6 SFT 后推理是否按 SETUP → STEP → ANSWER 顺序 commit"：

```python
# Patch serve.py / LLaDAWrapper 让每 step 记录被 commit 的 token 位置
# 对比 baseline vs T6 ckpt 的 commit 轨迹
# 看 T6 ckpt 是否 structural tokens 早 commit
```

TODO: 加一个 `--record_trajectory` flag 到 eval，log per-step commit 顺序，post-hoc 分析。

---

## 7. 后续方向（v1.7+ 候选）

- **显式 structural scheduler**：加一个 flag 让 scheduler 优先 commit `<SETUP>`/`<STEP_N>`/`<ANSWER>` 开始 tag，再填内容。类似 LogicDiff 但我们的结构是自定义的 tag 而非 classifier-predicted role
- **Section-block denoising**：不用 `block_length=32` 固定 layout，改成"按 structural section 划分 block"，每 section 独立 denoise
- **Canvas-position + order 组合**：inpainting 预钉 `<SETUP>` tag → 强制 section boundary → 配合 low-confidence 调度

都在 `docs/private/2026-04-19_unmasking_order_brainstorm.zh.md` 有展开。

---

## 8. Summary table

| 阶段 | Order 如何决定 | 显式/隐式 | 跟 baseline LLaDA 一致？ |
|---|---|---|---|
| Training mask | `rand() < t` random per-position | 隐式均匀 | ✓ 完全一致 |
| Inference block 间 | 左→右固定 | 显式 | ✓ 一致 |
| Inference block 内 | argmax confidence | 显式 | ✓ 一致 |
| Structural effect | 通过训练数据间接塑造 confidence | 隐式 | **不同** —— SFT 后 canvas 结构更被倾向生成 |
