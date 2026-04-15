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
- **Order axis**：T=0.0 固定，换 3 种 scheduler (`low_confidence`, `random_remask`, `cot_dag`) → 3 个 output
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
| H0 | `h0_forensics.py` | DONE | 137 fail prompts → runs/validation/scope_fail_prompts.json | 2026-04-15 |
| H1 | `h1_remask_rescue.py` | — | — | — |
| H2 | `h2_order_vs_content.py` | — | — | — |
| H3 | `h3_passN_at_temperature.py` | — | — | — |
