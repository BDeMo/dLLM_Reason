# Finding (N=60, offline): CoT-broken 和 A4-only-rescue 都没有手工特征信号

> 语言：中文
> 日期：2026-04-15
> 脚本：`scripts/validate/p4_cot_broken_pattern.py` · `scripts/validate/p6_a4_rescue_features.py`
> 输出：`runs/validation/p4_cot_broken_pattern_*.json` · `runs/validation/p6_a4_rescue_features.json`

---

## P4 —— CoT 模板砸 baseline-对的那些 prompt 的共性

跑 7 个 feature (char/word/sentence/digit/arith-word/arith-sym/how-hits) 的 Mann-Whitney U 检验（broken vs ok）。

**cot_step_broken (n=16) vs cot_step_ok (n=26)**
- 所有 feature **|z| < 1.64**（阈值粗对应 two-sided p<0.1）
- 最大 z：`sentence_count = -1.22`（broken 句子略少，不显著）

**cot_plain_broken (n=11) vs cot_plain_ok (n=31)**
- 所有 feature 同样不显著
- 最大 z：`arith_word = +1.50`（broken 有更多"twice/plus/per"等词，但不达阈值）

**结论**：这 7 个表层特征抓不到 CoT-broken 的模式。要么：
- (a) Pattern 在更深的结构层（prompt semantics / problem type），表层统计看不到
- (b) N=60 里 broken=16/11 太小，真有信号也被噪声吃掉
- (c) 就是没有结构化 pattern，纯 per-prompt variance

**行动**：等 N=137 再跑；如果 n_broken 翻到 ~35 仍全不显著，放弃 per-prompt template selector，坚持 `{baseline, answer}` ensemble 路线。

---

## P6 —— A4 独救条目的手工特征

分组：`a4_only` (n=2) / `shared` (n=3) / `a5_only` (n=5) / `no_rescue` (n=8)。

**统计结论**：`a4_only_vs_shared` 在 char_len/word_len 上 z=-1.73（a4_only 更短），但 **n=2 的样本统计基本无意义**，噪声主导。

**对 P6 的实际判断**：N=60 下 A4-only 只有 2 条，**任何手工 rule 都不可能从 2 条样本学出**。P6 在 N=60 是**纯早熟**。等 N=137 A4-only 至少扩到 5-8 条，再看。

---

## 对 A 轴策略的影响

1. **CoT 模板砸的现象还是谜**。A5 report 说 cot_step 砸 12 条，特征上看不出原因。这意味着 per-prompt 模板选择器（比 ensemble 更 efficient 的方案）**在 N=60 数据里无法设计**。
2. **A4.1（layout 预测器）仍然不能启动**。需要 N=137 先产出更多 A4-only 样本。
3. **`{baseline, answer}` ensemble 和 `{block_length} × {template}` 多格 ensemble** 仍然是 A 轴 rescue 的现实上线路线，不依赖 per-prompt selector。

---

## Post-closure 注记

P4 数据（N=60）：cot_step_broken(16) vs ok(26) — 7 个 feature 全 |z| < 1.64，无显著差异。最大 z: sentence_count = -1.22。
P6 数据（N=60）：a4_only(2) vs others — n=2 太小无意义。

两项分析在 N=60 下都**无法产出有意义结论**。这不是"没有 pattern"的强证据，而是"样本太小看不到"。

## 归档状态

- P4/P6 **OPEN**，等 N=137 数据再 rerun
- 不降结论阈值；特征集可以扩（加 problem-type classifier 的结果、解的预期步数、数字大小范围等），但都等 N=137 一起做
- 如果 N=137 下 n_broken 翻到 ~35 仍全不显著 → 放弃 per-prompt template selector，坚持 ensemble 路线
