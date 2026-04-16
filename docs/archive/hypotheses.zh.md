# Post-DAG Pivot — 假设登记簿

> 语言：中文  |  English: [hypotheses.md](hypotheses.md)

**上下文**：`finding_dag_search_zero_rescue.md` 证明 v1.5.3 的 static DAG greedy search 对 gsm8k 挽救率 = 0。本文登记"真正的瓶颈在哪"的候选假设，每条配一个可运行验证脚本 + 明确证据阈值。

**规则**：
- 一次只验证一条
- 脚本跑完产出 **structured JSON result**（`runs/validation/h{n}_*.json`），内含 verdict = `SUPPORTED | REJECTED | INCONCLUSIVE`
- 每条假设验证完立刻在本文底部 append 结论
- 失败案例集固定：`scope_fail_prompts.json`（先跑 `h0_forensics.py` 生成），所有后续实验复用

---

## 假设清单

### H0 (exploratory)：失败案例的错误模式可分类
**不是一个要证伪的假设**，而是为后续实验定义可控 scope：从 `episodes.db` 读 137 条 `correct=0` 的样本，按 error 类型分桶（early-step commit 错 / 晚期数值错 / 格式错 / 根本不会做）。

**产出**：`runs/validation/scope_fail_prompts.json`，含 prompt / gt / output / error_category。
**阈值**：无 verdict，只生成 artifact。

---

### H1：Commit-once-never-revise 是主要瓶颈
**断言**：若在 fail 案例上启用 **revise hook**（每步后对 conf < τ 的已 commit token 重采样），rescue rate 显著 > 0。

**脚本**：`scripts/validate/h1_remask_rescue.py`
**做法**：
- 拉 H0 的 fail 集（最多 N=50 条，控时间）
- 对每条 prompt 跑两次推理：
  1. baseline = `llada_generate(T=0, remasking=low_confidence)` 原生
  2. revise = 原生 + 每 `revise_every=8` 步对 conf < `revise_thresh=0.3` 的已 commit token 重新置回 mask
- 对比两次的 correctness

**Verdict 阈值**：
- `rescue_rate = (revise_correct ∧ ¬baseline_correct) / N`
- `rescue_rate ≥ 5%` → **SUPPORTED**（H1 成立）
- `rescue_rate ≤ 1%` → **REJECTED**
- 中间 → **INCONCLUSIVE**

---

### H2：T=0 + 双向 attention 让 unmask order 近乎无关
**断言**：同一 prompt 上，**改顺序 (DAG)** 的输出方差 ≪ **改内容采样 (温度)** 的输出方差。即 order axis 信号量远小于 content axis。

**脚本**：`scripts/validate/h2_order_vs_content.py`
**做法**：
- 在 fail 集取 K=20 条 prompt
- **Content axis**：同一 scheduler (`low_confidence`, 无 DAG)，T ∈ {0.0, 0.3, 0.7}，每 T 采 3 次 → 9 个 output
- **Order axis**：T=0.0 固定，同一 `low_confidence` scheduler，换 3 个 `block_length` ∈ {16, 32, 64}（换 block size 即重排跨步 commit 顺序） → 3 个 output
- 对每条 prompt 算 output 的 normalized edit distance var

**Verdict 阈值**：
- `order_var / content_var < 0.3` → **SUPPORTED**（H2 成立：order 信号弱）
- `order_var / content_var > 0.7` → **REJECTED**
- 中间 → **INCONCLUSIVE**

---

### H3：llada-instruct 在这 137 条上达到能力上限
**断言**：即使加大采样多样性 (temperature + N 重采样)，这些 prompt 的 pass@N 依然 ≈ 0。

**脚本**：`scripts/validate/h3_passN_at_temperature.py`
**做法**：
- fail 集取 K=30 条
- 每条 prompt × T ∈ {0.3, 0.7, 1.0} × N=8 次采样 → 算 pass@1 / pass@4 / pass@8
- 参考列：用同一 K 条 `init_ok` prompt 做对照

**Verdict 阈值**：
- fail 集 `pass@8 < 5%` 且对照 `pass@8 > 90%` → **SUPPORTED**（H3 成立：能力上限）
- fail 集 `pass@8 > 20%` → **REJECTED**（能力没到上限，多样性能救）
- 中间 → **INCONCLUSIVE**

---

### H4（备用）：Block-wise 采样的 block boundary 是错误注入点
**断言**：错误集中出现在每个 block 的前 k 个被 commit 的 token（高 confidence 陷阱）。
**脚本**：占位，等 H1/H2/H3 结果再决定是否做。

---

### A3：Span-level revise 比 single-token revise 更有效
**断言**：错误藏在**连续 span**里，不在单 token 上。revise hook 换成"window 平均 conf < τ → 整窗置回 mask"能救到 H1 (single-token τ) 救不了的案例。

**脚本**：`scripts/validate/a3_span_revise.py`
**做法**：`F.conv1d` 在已 commit 位上做 sliding window（默认 `window_size=4, revise_thresh=0.4, revise_every=8`），window mean < τ 则把该窗口内所有 committed 位置回 mask。
**Verdict 阈值**：同 H1。

### A4：Block layout 决定成败 —— 换切分能救
**断言**：默认 `block_length=32` 对部分 reasoning 结构不是最优；`{8, 16, 32, 64, short_then_long}` 里至少一种 layout 能救。

**脚本**：`scripts/validate/a4_block_rerank.py`
**做法**：每条 prompt 在 block_length ∈ {8, 16, 32, 64} 各跑一次 + 一个非均匀 layout（前 64 tokens 用 block=16，后 64 tokens 用 block=64）。记录 `any_layout_correct`。
**Verdict 阈值**：`any_layout_rescue_rate ≥ 5%` → SUPPORTED / ≤ 1% → REJECTED。

### A5：Prompt template 集成能救
**断言**：CoT 前缀或 "Answer:" 前缀能把输出分布推到能解的区域。

**脚本**：`scripts/validate/a5_prompt_template.py`
**做法**：每条 prompt 追加 {baseline / "\nLet's solve this step by step." / "\nStep 1:" / "\nAnswer:"} 之一。记 `any_template_correct`。
**Verdict 阈值**：同 A4。

---

### E1：A6 g160 增益来自"空间"还是"计算步数"
**背景**：A6 `gen_length=160` 在 fail 集上 rescue 20%，但 A6 把 `num_steps` 和 `gen_length` 捆绑（都等于 g 值）。如果 rescue 来自额外计算步数，那就是 latent reasoning 证据；如果只来自额外 token 位置，那 A6 只是 budget calibration，不构成 latent reasoning 证据。

**脚本**：`scripts/validate/e1_gen_vs_steps.py`
**做法**：在同一 60 条 fail 上跑 3 个配置：
1. **C_g128_s128**（baseline）
2. **A_g160_s160**（= A6 g160，空间↑ + 步数↑）
3. **B_g128_s160**（空间锁 128，只把 num_steps 加到 160）

若 A rescue > 0 但 B rescue ≈ 0 → **space-effect**（latent reasoning REJECTED）。
若 A ≈ B rescue > 0 → **compute-effect**（latent reasoning SUPPORTED）。

**Verdict 阈值**：
- `rescue_rate_stepsB ≥ 5%` → **SUPPORTED** (latent reasoning 成立)
- `rescue_rate_stepsB ≤ 1%` → **REJECTED** (latent reasoning 被排除)
- 中间 → **INCONCLUSIVE**

### E5：A6 g128 tail 是否被 token budget 物理截断
**背景**：最 trivial 的反方解释 —— g128 答案写不完，g160 只是给了更多 token 位置让答案写完，不需要任何语义解释。

**脚本**：`scripts/validate/e5_truncation_check.py`
**做法**：offline 分析 A6 run 的 `tails` 字段（每条 prompt × 每个 gen_length，out[-200:] 存盘）。启发式判定：answer marker 正则 / 句号收尾 / 数字收尾 / mid-word 结尾 → verdict ∈ {complete, truncated, maybe_truncated, ambiguous}。
**聚焦**：A6 独救 3 条 {0, 19, 51} 的 g128 tail。

**Verdict 阈值**：
- 3 条里 ≥2 条 truncated/maybe → **TRIVIAL_TRUNCATION**（A6 gain 是 budget 效应）
- ≤1 条 → **NOT_TRUNCATION**（gain 不能用截断解释）

---

## 验证顺序

1. **H0**（零成本，先跑出 scope） → `scope_fail_prompts.json`
2. **H1**（最便宜，直接告诉我们 pivot 方向是否对）
3. **H3**（如果 H1 被 reject，H3 用来区分"order 没用但 content diversity 有用"还是"模型真不会"）
4. **H2**（补充证据，不是关键路径）

---

## 结论板（自动更新）

> 由 `scripts/validate/aggregate_verdicts.py` 扫 `runs/validation/h{1,2,3}_*/summary.json` 的最新时间戳自动覆盖。
> 手动加注释请写在本表**上方**，表格内容每次 aggregate 都会被重写。

| 假设 | 脚本 | Verdict | 关键数字 | 日期 |
|---|---|---|---|---|
| H0 | `h0_forensics.py` | DONE | 60 fail prompts → runs/validation/scope_fail_prompts.json | 2026-04-16 |
| H1 | `h1_remask_rescue.py` | REJECTED | N=137  base=0  revise=0  rescued=0  broken=0  rescue_rate=0.00% | 2026-04-16 |
| H2 | `h2_order_vs_content.py` | REJECTED | N=20  content_var=0.256  order_var=0.176  ratio=0.754 | 2026-04-16 |
| H3 | `h3_passN_at_temperature.py` | REJECTED | n_fail=30  n_ok=26  fail_p@8=86.67%  ok_p@8=100.00% | 2026-04-16 |
| A3 | `a3_span_revise.py` | REJECTED | N=60  base=42  revise=42  rescued=0  broken=0  rescue_rate=0.00% | 2026-04-16 |
| A4 | `a4_block_rerank.py` | SUPPORTED | N=60  base(bl32)=42  any=47  rescue_rate=8.33%  [bl8=43 bl16=41 bl32=42 bl64=37 short_then_long=37] | 2026-04-16 |
| A5 | `a5_prompt_template.py` | SUPPORTED | N=60  base=42  any=50  rescue_rate=13.33%  [baseline=42 cot_plain=35 cot_step=30 answer=45] | 2026-04-16 |
| A6 | `a6_gen_length.py` | SUPPORTED | N=60  base(g128)=42  any=54  rescue_rate=20.00%  [g64=27 g96=36 g128=42 g160=49 g192=39 g256=40] | 2026-04-16 |
| A4x5 | `a4x5_joint.py` | — | N=60  base=42  joint_any=52  rescue_rate=16.67%  [bl8_baseline=43 bl8_answer=41 bl32_baseline=42 bl32_answer=45 bl64_baseline=37 bl64_answer=40] | — |
| E1 | `e1_gen_vs_steps.py` | REJECTED | N=60  C_g128_s128=42  A_g160_s160=49  B_g128_s160=42  rescue_longA=15.00%  rescue_stepsB=0.00%  a6_only_longA=1/3  a6_only_stepsB=0/3 | 2026-04-16 |
| E5 | `e5_truncation_check.py` | NOT_TRUNCATION | a6_only_g128_trunc=1/3  [idx0:mayb/✗  idx19:comp/✗  idx51:comp/✗] | 2026-04-16 |
