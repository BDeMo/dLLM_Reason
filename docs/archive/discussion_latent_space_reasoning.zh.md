# Discussion: A6 gen_length 发现是否解释了 Latent Space Reasoning？

> 语言：中文
> 日期：2026-04-16
> 状态：**讨论已结束 —— 被 E1+E5 联合 REJECTED**。保留本文作为"A6 发现当时是如何被解读的"的审计痕迹。
> **结论请看**：[`finding_e1_e5_rules_out_latent_reasoning.zh.md`](finding_e1_e5_rules_out_latent_reasoning.zh.md)
> 配套数据：[`empirical_rescue_per_prompt.zh.md`](empirical_rescue_per_prompt.zh.md) · [`finding_gen_length_sensitivity.zh.md`](finding_gen_length_sensitivity.zh.md)

---

**⚠ 2026-04-16 addendum**：E1 (`e1_gen_vs_steps.py`) 和 E5 (`e5_truncation_check.py`) 已完成。E1 证实 A6 gain **不来自** num_steps（B_g128_s160 rescue=0%）；E5 排除 trivial 截断。A6 的 +11.7pp 来自 write-space effect（explicit CoT 需要更多 token 位置），跟 latent reasoning 无关。下文的 Pros 章节（§2）在 E1 后应重读为"这些观察**与** latent reasoning 一致，但**也与** write-space 一致，E1 把二者区分开了"。

---

---

## 0. 背景：什么是 Latent Space Reasoning

Autoregressive LM 的推理**必须发生在 token space**：每步产出一个 token，推理链的每一步都被编码进可读的 token 序列（Chain-of-Thought）。如果模型需要 5 步推理，它必须生成至少 5 步对应的 tokens。

**Latent space reasoning** 假说认为：某些模型（特别是 diffusion LM）的推理可以发生在**不直接对应 token 的 hidden states 中**。模型可以在 latent space 里"思考"若干步，最终在 token space 上输出答案，中间步骤不一定被显式展开成 tokens。

相关工作：
- Coconut (Hao et al., 2024)：autoregressive LM + continuous thought tokens，证明 implicit reasoning 可以用连续 token 替代 CoT
- DDPM/Score-based 生成：每步 denoise 本身就是一步"implicit computation"
- Diffusion forcing (Chen et al., 2024)：variable-length denoising = adaptive computation depth

---

## 1. A6 的发现与 latent reasoning 的关联

### A6 核心数据

```
gen_length:  64    96    128   160   192   256
accuracy:    45.0  60.0  70.0  81.7  65.0  66.7
```

- g160（5 blocks × 32 tokens）是甜点，比 baseline g128（4 blocks）**+11.7pp**
- g192/g256 回落 —— **不是越长越好**

### 直觉连接

在 LLaDA 的生成过程中：
- `gen_length` 决定了模型有多少 **token 位置** 可以使用
- `num_steps = gen_length` 意味着每增加一个 block，模型多走 32 步 diffusion
- 每步 diffusion = 一次 full-sequence attention forward pass

**如果 reasoning 发生在 diffusion 步骤的 hidden states 中**（而不只是在最终 committed 的 token 中），那么 g160 比 g128 多的 32 步就相当于**多了 32 步 latent reasoning**。模型并不需要在那 32 个额外 token 位置里写出有意义的文字 —— 它只需要利用那 32 步 forward pass 的 hidden state 来更好地 refine 前面的 token。

---

## 2. 支持 latent reasoning 解释的证据（Pros）

### 2.1 gen_length 增益不来自"更长的文字输出"

如果增益来自"有更多 token 写推理链"（= explicit CoT reasoning），我们预期：
- CoT template（cot_step、cot_plain）应该跟 longer gen_length 有协同
- 但事实上 **cot_step 在 g128 下已经是最差的 template（30/60）**

A5 的反直觉发现（CoT 模板反砸）跟 latent reasoning 一致：**强制模型把推理展开成 explicit tokens 反而打乱了它在 latent space 里的 reasoning 路径**。模型"想"用 hidden states 推理，但 cot_step 前缀逼它把中间步骤写成文字，反而引入了 token-level commitment 错误。

### 2.2 g160 的增益在 A5 里找不到对应物

- A5 any-template=50/60 → 说明 template 多样性能救 8 条
- A6 any-length=54/60 → gen_length 多样性能救 12 条（**比 A5 多 4 条**）
- A6 独有 3 条（idx=0,19,51）—— **只有 gen_length 变了才能拿到**，换 template 无用

这 3 条 A6 独有 prompt 的特征：
- idx=0 gt=70000 —— 大数计算
- idx=19 gt=18 —— 四步速度推理（Dana runs/walks/skips/crawls）
- idx=51 gt=9360 —— 多员工月薪 + 加薪比例

共性：**都需要多步算术但 gt 本身不复杂**。模型"知道怎么做"（在某个 gen_length 下做对了），但在 g128 下 latent computation budget 不够。

### 2.3 block-wise generation = discrete "thinking steps"

LLaDA 的 block-wise unmasking 跟 autoregressive 不同：
- Autoregressive：每步 commit 1 token，token_n 的 hidden state 只能看 token_0..n-1
- LLaDA：每步 commit **一整个 block**（32 tokens），所有 token 位置都能 attend 到所有其他位置（双向 attention）

这意味着 **每一步 diffusion 的 hidden state 是 global 的** —— 模型在每步 forward pass 里可以"同时看到整个序列的当前状态"。增加 gen_length = 增加 block 数 = 增加 global forward pass 数 = **增加 latent reasoning 步数**。

### 2.4 g192/g256 回落 = over-thinking

如果 latent reasoning 解释正确，g192/g256 的回落就是"over-thinking"：
- 多余的 diffusion 步骤引入了额外的 token commitment（模型必须在额外位置填东西）
- 这些被迫填入的 token 可能污染 context，导致后续步骤的 hidden state 偏移
- 类似 explicit CoT 里"思考太多反而错"的现象

这跟 CoT 的 "chain-of-thought 并不是越长越好" 一致（Wei et al., 2023 观察到 GSM8K 上过长的 CoT 会降性能）。

---

## 3. 反对 latent reasoning 解释的证据（Cons）

### 3.1 num_steps 和 gen_length 同时变了 —— 混淆因素

A6 的设计是 `num_steps = gen_length`。g160 比 g128 同时多了：
- 32 个 token 位置（**空间**）
- 32 步 diffusion（**计算**）

如果增益主要来自**空间**（"g128 写不下推理链，g160 刚好够"），那解释就是 trivial 的 explicit reasoning，跟 latent reasoning 无关。

**控制实验（尚未做）**：
```
(a) gen_length=160, num_steps=128   → 空间大，步数不变
(b) gen_length=128, num_steps=160   → 空间不变，步数多
(c) gen_length=160, num_steps=160   → 两者都变 (= A6 当前)
```
- 如果 (a) ≈ (c) >> (b) → 纯空间效应，不支持 latent reasoning
- 如果 (b) ≈ (c) >> (a) → 纯计算效应，**强支持** latent reasoning
- 如果 (a) ≈ (b) ≈ (c) → 两者等价，无法区分

### 3.2 g160 输出**确实比 g128 更长**

如果模型在 g160 下只是"写了更多 explicit 推理步骤"就做对了，那增益就是 trivial 的 —— 不需要 latent reasoning 来解释。

验证方法：看 g160 vs g128 的输出长度和内容。如果 g160 的输出多了一步 explicit 算术（比如 g128 漏掉了 "Step 3: 70000+5000=75000"，而 g160 写出来了），那就是纯 explicit reasoning。

**已有数据**（P2.1.e dump）只覆盖 5 条，且是 g256 不是 g160。需要专门对 A6 独有的 3 条 (idx=0,19,51) 做 g128 vs g160 的输出对比。

### 3.3 LLaDA 的 attention 是双向的 —— 但 diffusion 步骤是有方向的

虽然 LLaDA 的 attention 是双向的，但 block-wise unmasking 引入了**时间方向**：先 commit 的 block 锁定了 context，后 commit 的 block 在这个 context 上生成。这跟 autoregressive 的"从左到右"在结构上相似。

这意味着 "latent reasoning" 如果存在，它的 pathway 不是"一个 forward pass 内的 layer-by-layer computation"（那是任何 transformer 都有的），而是**跨 diffusion 步骤的 hidden state 演化**。这是一个更强的声称，需要更强的证据。

### 3.4 Diffusion LM 的 "implicit computation" 不等于 "reasoning"

DDPM 每步 denoise 确实做了 implicit computation，但这不等于 "reasoning"。图像 diffusion 每步 denoise 也在做 implicit computation（边缘锐化、纹理细化），但没人说 DDPM 在"推理"。

区分标准：如果 gen_length 的增益**只在需要多步推理的 prompt 上出现**（比如 idx=0,19,51 都是多步题），而在单步题上没有增益，那支持 "latent reasoning"。如果在所有题上均匀提升，那更可能是 "more compute = better denoising"。

### 3.5 可能的 trivial 解释："g128 = 太短，很多答案被截断"

最无聊的解释：g128=128 tokens 对某些 prompt 的答案来说**物理上写不完**。如果答案需要 150 tokens，g128 必然错（被截断），g160 刚好够。

检查方法：看 g128 输出是否以不完整的句子/数字结尾（被截断的标志）。如果是 → gen_length 增益是 trivial 的截断问题，完全不需要 latent reasoning 来解释。

---

## 4. 可做的区分实验

| 实验 | 目的 | 预期结果 if latent reasoning | 代价 |
|---|---|---|---|
| **(E1) 拆分 gen_length vs num_steps** | 区分空间 vs 计算 | (b) >> (a)，即步数增益大于空间增益 | 中（改 serve.py 支持 gen≠steps） |
| **(E2) g128 vs g160 输出对比** | 看增益是 explicit 多步 or not | g160 的输出**不显著更长/更详细**，但答案更对 | 低（已有数据可查） |
| **(E3) 按 prompt 难度分层分析** | 增益集中在多步题？ | rescue 只在 ≥3 步题上出现 | 低（标注 prompt 步骤数） |
| **(E4) hidden state 可视化** | 直接观测 latent trajectory | 不同 gen_length 的 hidden trajectory 收敛性不同 | 高（需改 model 代码） |
| **(E5) g128 截断检查** | 排除 trivial 解释 | g128 输出不以截断结尾 | 极低（读 tail） |

**最高优先**：E5 → E2 → E1。如果 E5 发现截断是主因，整个 latent reasoning 讨论就不成立。

---

## 5. 更大的图景：Diffusion LM 作为 "Thinking Machine"

如果 latent reasoning 假说成立（即 diffusion 步骤确实承载了 implicit reasoning），那么我们的实验数据给出了一个比 Coconut 更自然的 latent reasoning 实例：

| 维度 | Coconut | LLaDA (ours) |
|---|---|---|
| Architecture | Autoregressive + continuous thought tokens | 原生 discrete diffusion |
| Latent space | 显式引入 continuous token 做 proxy | **diffusion 步骤本身就是 latent computation** |
| Thinking budget | 手动设定 thought token 数量 | **gen_length / num_steps 隐式控制** |
| 证据 | 在 ProntoQA/GSM8K 上 latent > explicit CoT | g160 > g128 且 CoT template 反砸 |
| 局限 | 需要修改训练 pipeline | **零训练，纯 inference-time discovery** |

如果 E1 确认增益来自 num_steps（不是 gen_length 空间），那这就是 **"discrete diffusion LM 天然支持 adaptive latent reasoning，只需调 inference budget"** 这个 claim 的第一个 empirical 证据。

---

## 6. 对 strategy search 的影响

不管 latent reasoning 是否成立，实操层面的 implication 是：

1. **gen_length 应该作为 strategy search 的核心维度**（不是 optional 旋钮）
2. **num_steps 可能需要跟 gen_length 解耦**（E1 实验），成为独立维度
3. **CoT template 可能有害** —— 如果 latent reasoning 是对的，per-prompt strategy 应该 bias 向"短 template + 长 gen_length"而不是"CoT + 短 gen_length"
4. **training-side distill（Phase 3）如果做 SFT，不应该强制 CoT 格式** —— 让模型自己选择在 token space 还是 latent space reasoning

---

## 7. 结论

A6 的 g160 甜点、A5 的 CoT 反砸、A4 的 block-layout 信号，三者组合起来**与 latent space reasoning 假说一致**，但**不构成证明**。

核心歧义：gen_length 同时改变了空间和计算，必须做 E1 实验拆分。如果 E1 确认增益来自 num_steps（计算步数），那这组实验就从 "inference strategy tuning" 升级成 **"diffusion LM 存在 latent reasoning 的 empirical evidence"**，可以写进 paper 的核心叙事。

**建议**：把 E1（gen_length vs num_steps 拆分）提升到跟 strategy search 同等优先级。它决定了这个项目的 paper story 是 "better inference strategy" 还是 "latent reasoning in diffusion LM"。后者显然更有影响力。
