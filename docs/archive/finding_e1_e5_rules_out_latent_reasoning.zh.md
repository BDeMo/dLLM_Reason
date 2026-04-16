# Finding: E1 + E5 联合排除 latent space reasoning 解释

> 语言：中文
> 日期：2026-04-16
> 状态：**REJECTED (latent reasoning as explanation for A6 gain)**
> 来源：`scripts/validate/e1_gen_vs_steps.py` + `scripts/validate/e5_truncation_check.py`
> 配套讨论：[`discussion_latent_space_reasoning.zh.md`](discussion_latent_space_reasoning.zh.md) · [`finding_gen_length_sensitivity.zh.md`](finding_gen_length_sensitivity.zh.md)

---

## TL;DR

A6 `gen_length=160` 在 fail 集上 rescue 15–20%，原本有两种可能解释：

1. **latent reasoning**：`num_steps=160` 给了模型多 32 步 diffusion forward pass，在 hidden states 里做了隐式推理
2. **write-space effect**：多 32 个 token 位置让 explicit CoT 铺得开

E1 解耦 `gen_length` 和 `num_steps`，E5 做截断启发式检查。联合结论：

- **E5 (NOT_TRUNCATION)**：A6 独救 3 条的 g128 tail 不是物理截断，都是 reasoning 错误
- **E1 (REJECTED)**：单加 `num_steps` 零贡献、零破坏 —— LLaDA 的 block-wise denoising 在 128 步已饱和

→ **A6 的 gain 来自空间而非计算**，latent reasoning 假说对 A6 无解释力，A6 只是 explicit CoT 的 space-budget calibration。

---

## 1. E5：先排除 trivial 截断解释

### 设计

`discussion_latent_space_reasoning.zh.md` §3.5 指出最无聊的反方解释：g128=128 tokens 对某些 prompt 物理写不完答案。如果 A6 独救 3 条 `{0, 19, 51}` 的 g128 tail 都是"截断到一半"，那 A6 增益完全不需要 latent reasoning 解释。

E5 对 A6 run 的 `tails` 字段（`out[-200:]`）做 offline 启发式判定：answer marker 正则 / 句号收尾 / 数字收尾 / mid-word 结尾 → verdict ∈ `{complete, truncated, maybe_truncated, ambiguous}`。

### 结果

A6 独救 3 条的 g128 tail 判定：

| idx | gt | g128 verdict | 答案 tail | 正确？ |
|---|---|---|---|---|
| 0 | 70000 | complete | "50000" | ✗ reasoning error |
| 19 | 18 | maybe_truncated | "24" | ✗ reasoning error |
| 51 | 9360 | complete | "726" | ✗ reasoning error |

**Verdict: NOT_TRUNCATION** （1/3 maybe，0/3 明确 truncated）。

### 解读

A6 独救 3 条在 g128 下都写出了"完整的、带数字的、但错误的"答案。不是空间不够写答案 —— 是**用 128 tokens 的空间走了一条错的推理链**。g160 的 rescue 必然来自别的机制（更多空间让模型走对另一条推理链），而不是 "终于写下答案了"。

E5 没有证伪 latent reasoning，但**排除了最 trivial 的反方解释**。这让 E1 的设计问题变成真正有意义的："多出来的 32 步是空间效应还是计算效应？"

---

## 2. E1：解耦空间与计算

### 设计

`discussion_latent_space_reasoning.zh.md` §3.1 §4 提出的 E1 实验。A6 把 `gen_length` 和 `num_steps` 捆绑（都等于 g 值），必须拆成：

| Config | gen_length | num_steps | 含义 |
|---|---|---|---|
| **C_g128_s128** | 128 | 128 | baseline |
| **A_g160_s160** | 160 | 160 | = A6 g160（空间↑ + 步数↑） |
| **B_g128_s160** | 128 | 160 | 空间锁 128，只加步数 |

判定规则：
- A rescue > 0 ∧ B rescue ≈ 0 → **space-effect**（latent reasoning REJECTED）
- A ≈ B rescue > 0 → **compute-effect**（latent reasoning SUPPORTED）
- A ≈ B ≈ 0 → INCONCLUSIVE

### 结果（N=60 fail set）

```
per_config_correct:
  C_g128_s128 = 42 / 60
  A_g160_s160 = 49 / 60   ← 比 C +7
  B_g128_s160 = 42 / 60   ← 比 C ±0

rescue vs baseline C:
  rescue_rate_longA  = 9/60 = 15.0%   (空间↑)
  rescue_rate_stepsB = 0/60 = 0.0%    (步数↑)

broken vs baseline C:
  broken_by_longA  = 2
  broken_by_stepsB = 0

A6 独救 3 条 {0, 19, 51} focus:
  idx=0   C=✗  A=✗  B=✗
  idx=19  C=✗  A=✓  B=✗    ← 关键点：空间起作用，步数不起作用
  idx=51  C=✗  A=✗  B=✗
```

**Verdict: REJECTED** (latent reasoning ruled out)。

### 解读

1. **B 配置（只加步数）rescue 率严格 0%**。不是"微小提升但未达阈值"，是**精确相等**的 42 vs 42：B 输出跟 C 输出在 correctness 上完全一致。
2. LLaDA block-wise denoising 在 `num_steps = gen_length = 128` 时已经饱和 —— 多出来的 32 步对最终 token 分布**零影响**。这跟 latent reasoning 假说预期的"多步 diffusion 让 hidden state 继续细化"完全相反。
3. idx=19 只在 A 配置下被救 —— 这条 prompt 需要"更多空间"而不是"更多计算"。H3 之前救不了 idx=19 原本被解读为"latent reasoning 正面证据"，E1 现在把它重新解读为"**explicit CoT space requirement** 不可被 diversity 替代"，跟 latent reasoning 无关。
4. `broken_by_stepsB = 0` 进一步印证：B 配置甚至没有"扰动"baseline。这强烈暗示 num_steps 从 128→160 在 LLaDA 的采样动力学里是惰性操作。

---

## 3. 对 `discussion_latent_space_reasoning.zh.md` 各 Pro/Con 的回应

### Pros 的新解读

| Pros | 原解读（支持 latent reasoning） | E1+E5 后解读 |
|---|---|---|
| §2.1 CoT template 反砸 | latent reasoning 一致：CoT 逼出 explicit commit 污染 latent 路径 | 还是一致，但跟 latent 无关 —— CoT template 本身改变了 prompt 分布，效果独立于 gen_length 机制 |
| §2.2 A6 独有 3 条 换 template 无用 | 这些 prompt 需要 latent compute | **重新归因为 write-space**：这些 prompt 需要更多 explicit CoT 空间 |
| §2.3 block-wise = 离散 thinking steps | 增 block 数 = 增 latent reasoning 步数 | 但加步数不改 correctness (B=C)，所以 latent 步数增加对 correctness 零贡献 |
| §2.4 g192/g256 回落 = over-thinking | over-thinking 假说 | 也可以是 "space 太多引入无意义 token，污染 decoder" —— over-decoding 而非 over-thinking |

### Cons 现在状态

| Cons | 状态 |
|---|---|
| §3.1 num_steps 和 gen_length 混淆 | **已拆解**：E1 证实增益来自 gen_length 这一维 |
| §3.2 g160 输出可能单纯更长 | 部分验证：A 配置 rescue 但 B 不 rescue，跟 "explicit 多写一步" 一致 |
| §3.3 双向 attention ≠ 时间无方向 | 未动，但不再重要（latent 已被排除） |
| §3.4 implicit computation ≠ reasoning | 未动，但不再重要 |
| §3.5 g128 截断 | **已排除**（E5） |

### 区分实验表

| 实验 | 状态 |
|---|---|
| E1 | **DONE** → REJECTED |
| E2 g128 vs g160 输出对比 | 可选（E1 已足以下结论，E2 只是 "确认是 explicit CoT 多一步"） |
| E3 按难度分层 | 可选，不影响结论 |
| E4 hidden state 可视化 | **取消**（latent 已被排除，不需要了） |
| E5 截断检查 | **DONE** → NOT_TRUNCATION |

---

## 4. 对论文 story 的影响

`discussion_latent_space_reasoning.zh.md` §7 建议：
> 如果 E1 确认增益来自 num_steps（计算步数），这组实验就从 "inference strategy tuning" 升级成 "diffusion LM 存在 latent reasoning 的 empirical evidence"。后者显然更有影响力。

E1 的答案是 **"增益不来自 num_steps"**。所以论文 story 回落到 "inference strategy tuning" 分支。具体定位：

### 不写的 story（被 E1 排除）
- ❌ "diffusion LM 天然支持 adaptive latent reasoning"
- ❌ "num_steps 是 thinking budget 的隐式旋钮"
- ❌ "LLaDA 在 hidden states 里做隐式推理"

### 保留的 story（E1+E5 后仍然成立）
- ✅ "gen_length 是最强单轴 rescue 信号，+11.7pp on gsm8k fail set"
- ✅ "per-prompt strategy search 给出 95.41% oracle 上限（vs baseline 83.49%）"
- ✅ "block_length / template / gen_length 存在 prompt-specific optimum，且不同轴彼此互补（A6 独有 3 条、A5 独有 2 条）"
- ✅ "diffusion LM 的 inference budget 分布是异构的 —— 不同 prompt 需要不同量的 explicit CoT 空间"
- ✅ "LLaDA block-wise denoising 在 `num_steps = gen_length` 时饱和" —— 这本身是个 publishable 的小发现，意味着加 num_steps 是浪费算力

### 核心叙事改写

| 维度 | 原叙事（latent reasoning） | 新叙事（space calibration） |
|---|---|---|
| 主 claim | diffusion LM 能在 hidden states 做推理 | per-prompt CoT space requirement 是异构的，固定 gen_length 不适合所有 prompt |
| 关键数字 | "+11.7pp 来自 32 步 latent compute" | "+11.7pp 来自 32 token 的额外 CoT canvas" |
| 机制叙述 | 32 步多 forward pass 让 hidden state 细化 | 32 个多位置让推理链写得下 |
| 可推广性 | 其他 diffusion LM 也应该有 | 其他 autoregressive LM 也有（gen_length tuning 早就存在） |
| 新颖性 | 高（latent reasoning empirical evidence） | 中（strategy search + budget heterogeneity） |

新叙事的新颖性低于 latent reasoning，但**扎实**。不需要 overclaim。

---

## 5. 对 strategy search 的影响

E1 REJECTED 把 strategy search 的维度设计简化了：

1. **`num_steps` 不作为独立维度**：E1 证实加 num_steps 零收益、零破坏。strategy search 固定 `num_steps = gen_length`（沿用 A6 约定），节省一个维度。
2. **`gen_length` 保持核心维度**：+15–20% rescue 是最强信号，必须搜。
3. **CoT template 维度改解读**：不再是"template 打乱 latent 路径"，而是"template 改变 prompt 分布让输出走向不同 attractor"。策略不变，解读变。
4. **Phase 3 SFT distill**：不受影响 —— 不管机制是什么，per-prompt 最优配置是可 distill 的。

---

## 6. 附：关键数字速查

```
A6 g160 vs g128 (baseline):
  accuracy delta  = +11.7pp (fail set), +6.42pp (full gsm8k)
  rescue          = 12 / 18 base_fail

E1 (N=60 fail set):
  C_g128_s128     = 42 / 60
  A_g160_s160     = 49 / 60  (= A6 g160)
  B_g128_s160     = 42 / 60
  rescue_longA    = 15.0%    ← 空间效应
  rescue_stepsB   = 0.0%     ← 计算效应（零）
  broken_longA    = 2
  broken_stepsB   = 0

E5 (A6 run offline):
  a6_only_idx            = {0, 19, 51}
  a6_only_g128_truncated = 1/3 maybe, 0/3 明确 truncated
  verdict                = NOT_TRUNCATION
```

---

## 7. 下一步

- **已完成**：E1 + E5 联合排除 latent reasoning
- **归档**：本文档 + hypotheses 板 + empirical_rescue_per_prompt 的 §6 E1+E5 联合结论
- **后续**：strategy search Phase 1 按 "space calibration" 叙事跑；Phase 3 SFT distill 不受影响
- **不做**：E2（g128 vs g160 输出对比）、E4（hidden state 可视化） —— E1 已足以下结论
