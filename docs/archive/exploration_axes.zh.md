# Post-DAG 探索方向索引

> 语言：中文  |  English: [exploration_axes.md](exploration_axes.md)

**上下文**：DAG search 三次独立实现（greedy / NAS supernet / E2E differentiable，见 `finding_dag_search_zero_rescue.zh.md`）全部 0 rescue；第一批假设 H1（token-revise）和 H2（order-vs-content）也返回 REJECTED。现在需要一张"下一步往哪挖"的地图。

本文把探索方向分成两根正交轴：
- **A 轴 —— 粒度梯度**：从细到粗扫一遍，只改变"干预"的作用单元。
- **B 轴 —— 正交方向**：改 inference sampler 之外的东西（训练 / 工具 / verifier）。

**约定**：每项带一个状态标签 —— `DEAD`（证伪）/ `PLANNED`（未跑）/ `RUNNING`（在跑）/ `DONE`（已记入 `hypotheses.zh.md`）。只有 N ≥ 30 + 有 verdict 才能打 `DEAD`。

---

## A 轴 —— 粒度梯度（从细到粗）

A1/A2/A3 已 DEAD/REJECTED；A4 **SUPPORTED**（block-layout 8.33% rescue，N=60）；A5 **SUPPORTED**（prompt-template 13.33% rescue，N=60）；A6 **SUPPORTED**（gen-length 20.00% rescue，N=60，最强单轴信号）。信号从 block 级开始出现，一直延伸到 prompt/gen-length 级。A4x5 joint 6-cell 实跑完美验证 overlap 预测（rescue=10=10）。

### A1 —— edge-level DAG rewiring · **DEAD**
- 证据：greedy ±1 edge（1319 prompts, 0 rescue, ~3072 edges 被搜）+ NAS supernet（200/0, 0 edges 选中）+ E2E differentiable（106/0, 0 edges 选中）。三个独立 optimizer 独立达到 0。
- 结论：T=0 + 双向 attention 下，edge-level 顺序对 greedy low-confidence 基线几乎零信号。

### A2 —— single-token revise hook · **DEAD**
- 证据：H1 在 137 fail prompts 上 rescue_rate=0；其中 122/137 revise hook 根本没触发（conf 一直 ≥ 0.3）。
- 结论：per-token confidence 不是 fail-prompt 上可靠的错误信号 —— "自信但错"是常态。

### A3 —— span-level revise · **REJECTED**
假设：错误藏在**连续 span**里（比如一个算错的小算式），单个 token conf 高，但 window 平均 conf 低。
- 脚本：`scripts/validate/a3_span_revise.py`
- 做法：已 commit 位上做 sliding window（默认 `window_size=4`），`F.conv1d` 算 mean conf；mean < τ（默认 0.4）则把**整个窗口的 committed 位**置回 mask。对照 H1 的 single-token hook。
- 结果 (N=60)：base=42, revise=42, rescued=0, broken=0, `rescue_rate=0.00%` → **REJECTED**。
- 启示：配合 H1（token 级 revise REJECTED），在 block 以下的任何粒度，confidence 信号都不含可用的错误信息。

### A4 —— block-layout rerank（吸收旧 A6）· **SUPPORTED**
假设：固定 `block_length=32` 并不一定是最优 layout；换一种切分（reasoning 段用短 block，answer 段用长 block；或单纯换均匀 size）能救一些 error。
- 脚本：`scripts/validate/a4_block_rerank.py`
- 做法：每条 fail prompt 在 block_length ∈ {8, 16, 32, 64} 各跑 1 次 + 一个非均匀 layout（短-长，模拟"推理步→最终答案"）。算 `any_layout_correct`。
- 结果 (N=60)：base(bl32)=42, any_layout=47, rescued=5, `rescue_rate=8.33%` → **SUPPORTED**。
- 启示：H2 半暗示的"order 信号"活在 **block 边界**，不在 edge（A1 DEAD）也不在 token（A2/A3 DEAD）。block-layout 是第一根有真实 rescue 信号的轴。

### A5 —— prompt-template rerank · **SUPPORTED**
假设：gsm8k 原 prompt 形状卡住了分布；CoT 前缀或直接 "Answer:" 前缀能把分布推到能解的区域。
- 脚本：`scripts/validate/a5_prompt_template.py`
- 做法：每条 fail prompt 用 4 个 template（baseline / "\nLet's solve this step by step." / "\nStep 1:" / "\nAnswer:"）各跑 1 次。算 `any_template_correct`。
- 结果 (N=60)：base=42, any_template=50, rescued=8, broken=0, `rescue_rate=13.33%` → **SUPPORTED**。
  - per-template：`baseline=42, cot_plain=35, cot_step=30, answer=45`。
- 启示：**反直觉** —— 单模板最好的是 `answer`（比 baseline +3），两个 CoT 模板反而**砸**（`cot_plain` −7、`cot_step` −12）。rescue 信号来自*模板多样性*，不是 CoT 本身。LLaDA-instruct 在 gsm8k 上已经被调成直接给答案，prompt 层面强塞 step-by-step CoT 反而打乱更多 prompt。

### A6 —— gen-length rerank · **SUPPORTED**
假设：默认 `gen_length=128` 不是所有 prompt 的最优生成长度；不同长度能救一些 error。
- 脚本：`scripts/validate/a6_gen_length.py`
- 做法：每条 fail prompt 在 gen_length ∈ {64, 96, 128, 160, 192, 256} 各跑 1 次（固定 block_length=32）。算 `any_length_correct`。
- 结果 (N=60)：base=42, any_length=54, rescued=12, `rescue_rate=20.00%` → **SUPPORTED**。**A 轴最强单轴信号**。
  - per-length：`g64=27, g96=36, g128=42, g160=49, g192=39, g256=40`。
  - 甜点：g160=49/60=81.7% vs baseline g128=42/60=70%。
- 启示：gen_length 是 A 轴最大的旋钮。g160 单点已超过 A5 的 any-template ensemble。但 g64/g96 明显差，说明不是"越长越好"，而是有最优长度。

### A4×A5 joint 6-cell 实跑验证

`{baseline, answer} × {bl8, bl32, bl64}` 6 格配置实跑结果：
- N=60, base=42, any=52, rescued=10, rescue_rate=**16.67%**
- **完美验证 overlap 预测**：预测 10 条 rescue，实跑 10 条，100% 吻合，零意外。
- per-cell: `bl8_baseline=43, bl8_answer=41, bl32_baseline=42, bl32_answer=45, bl64_baseline=37, bl64_answer=40`

---

## B 轴 —— 正交方向

A 轴扫完，B 轴评估更新。

### B1 —— pass@N diversity sampling · **最强单维度杠杆（= H3, n=60）**
H3 (`h3_passN_at_temperature.py`) 已扩到 n=60 完整 fail 集重跑（`h3_passN_20260415_133254`）。

**权威结果（n=60 fail）**：
- `fail_pass@8_max = 86.67%` (T=1.0)，`ok_pass@8_max = 100%` —— 按 hypotheses.md capacity-ceiling 阈值 → REJECTED（= 能力没到上限）
- P6 crossref rescue 口径：H3 rescue = **52/60 (86.67%)**，H3 stuck = **8 条 [4,5,14,19,41,42,48,51]**
- 只 H3 救到的（A 轴都没救）= **42 条**；H3 ⊆ A-union 仅 **19.2%**（与 A 轴近乎正交）

H3 在 n=60 下**远超 A 轴全体**（A-union = 13/60 = 21.67%）。旧 n=30 初版的 "7/30 = 23.33%" 是偏小样本数字；新 n=60 下 pass@N 是**最强单维度杠杆**，独立于 A 轴。

trade-off：H3 要付 24× 推理代价（3 temps × N=8），A-union 只要 1× 但上限 21.67%。往 paper story 上推，如果 inference cost 不是 dealbreaker，pass@N 是最有效的 rescue 方式。

### B2 —— training-side pivot
SFT on 137 fail prompts（从强 solver 蒸馏）或 correctness RL。脚本未写，等时机到了放 `scripts/train/`。

### B3 —— tool-augmented eval
Inference 时接 calculator / Python executor；度量 gsm8k 错误里多少是纯算术、多少是推理结构。脚本未写。

### B4 —— verifier / critic head
加一个轻量 head 对最终答案打分；single-pass self-correction。脚本未写。

---

## 路由逻辑

```
A3 REJECTED   —— conf-based revise 在 token/span 粒度全死
A4 SUPPORTED  —— block-layout rerank → 8.33% rescue
A5 SUPPORTED  —— prompt-template rerank → 13.33% rescue
A6 SUPPORTED  —— gen-length rerank → 20.00% rescue（A 轴最强单旋钮）
A4x5 joint    —— 6-cell 实跑 16.67%，完美验证 overlap 预测
H3 (n=60)     —— pass@8 = 86.67% rescue（capacity ceiling REJECTED；最强单维度杠杆）

Rescue 集交叉（n=60 权威, P6 crossref）：
  A4∪A5 = 10, A4∪A5∪A6 = 13, A-union = 13/60 (21.67%)
  H3 = 52/60 (86.67%)，H3 ∩ A-union = 10，H3 ⊆ A-union = 19.2%
  全 union = A-union ∪ H3 = 55/60 = 91.67%
  true capacity ceiling = 5/60 [4,5,14,41,42]
  axis-only: h3_only=42, a6_only=[19,51], a4_only=a5_only=[]
```

**下一步优先级 —— per-prompt strategy search pipeline**：

A 轴已充分扫完，全方法 union **91.67% (55/60)** 说明 inference-time 旋钮空间足够大。下一阶段从"哪个轴有信号"转到"对每条 prompt 找最优配置"：

1. **Per-prompt strategy search**：对每条 prompt 搜最优 `(block_length × template × gen_length × temperature)` 组合，存 `(prompt, best_strategy)` pairs。
2. **新维度 template_position**：diffusion LM 特有的 scaffold/inpainting —— 把 template token 放在生成区域任意位置（不只是 suffix）。这是 AR LM 做不到的。
3. **训模型学策略**：用 strategy search 的 pairs 训一个 `prompt → best_strategy` 预测器，让模型学会这些策略。
4. **P4/P6 离线分析**：等 N=137 再跑（当前 N=60 下 7 个 feature 全不显著）。
