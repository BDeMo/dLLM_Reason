# Plan: A 轴探索阶段（2026-04-15）

**日期**：2026-04-15（首版）/ 2026-04-16（closure）
**状态**：done（retrospective 补记）
**前置上下文**：A1 edge DAG 三证 0 rescue（archived in `finding_dag_search_zero_rescue.zh.md`）
**继任**：[2026-04-16_strategy_search_design.zh.md](2026-04-16_strategy_search_design.zh.md)

## 目标

在 137 条 gsm8k fail prompt 上，用一系列递增粒度的 inference-time 干预逐个实验，回答"LLaDA-Instruct 的可救 rescue 信号在哪个粒度"。

## 决策 / 方法

按"粒度阶梯"顺序扫：

1. **H1 / A2**：token-level revise hook（τ=0.3）—— 预期能救
2. **A3**：span-level window revise（τ=0.4, window=4）—— H1 死了再试宽阈值
3. **H2**：order var vs content var —— 理论先验
4. **A4**：block-layout rerank（5 layouts）—— H2 指向的粒度
5. **A5**：prompt template rerank（4 templates + n=60 5 templates）
6. **A6**：gen-length rerank（6 lengths）
7. **A4×A5 joint 6-cell**：ensemble overlap 验证
8. **H3**：pass@N capacity ceiling（T ∈ {0.3, 0.7, 1.0}, N=8）

每个实验预设阈值 verdict（`rescue_rate ≥ 5%` → SUPPORTED）。

## 预期输出

- 每个实验一个 finding doc
- 统一的 hypotheses.zh.md verdict board
- 跨实验 cross-ref（P5 → P6 fixed crossref）

## Updates

- **2026-04-15**: H1/A3/H2 全 DEAD 或 REJECTED。Token/span/edge 级完全没信号。
- **2026-04-15**: A4 SUPPORTED（5/60=8.33%），bl8 > bl32 >bl64；short_then_long 破坏多条。
- **2026-04-15**: A5 SUPPORTED（8/60=13.33%），`answer_marker` 单独 beat baseline，CoT 前缀反而砸。
- **2026-04-15**: A4×A5 joint 6-cell 预测 10 条 rescue 实跑 10 条，完美验证 overlap。
- **2026-04-16**: A6 SUPPORTED（12/60=20%），g160 是甜点，**A 轴最强单旋钮**。独家救 {19, 51}。
- **2026-04-16**: H3 小样本 n=30 得 7/30=23.33% SUPPORTED 初报 → **误读**。
- **2026-04-16**: H3 扩到 n=60 = **52/60 = 86.67%**，REJECTED per capacity-ceiling 阈值（能力远没到上限）。
- **2026-04-16**: P5 crossref silent bug，改写为 P6。P6 权威输出：full_union = 55/60 = 91.67%，ceiling 5 = `{4, 5, 14, 41, 42}`，a6-only = `{19, 51}`，h3-only(FAIL18 外)=42。

## Retrospective

### 实际做了什么

全部 8 个实验跑完。归档 docs 沿时间线更新：
- `finding_dag_search_zero_rescue.zh.md`（A1 收尾）
- `finding_a_axis_exploration.zh.md`（A2-A6 + H3 主叙事）
- `finding_a4x5_overlap.zh.md`（joint 6-cell）
- `finding_gen_length_sensitivity.zh.md`（A6 详）
- `finding_e1_e5_rules_out_latent_reasoning.zh.md`（E1/E5 rule out latent reasoning）
- `empirical_rescue_per_prompt.zh.md`（FAIL18 逐条表）
- `closure_a_axis.zh.md`（A 轴整体 closure）

### 主要发现（paper 可 cite 的）

1. **Granularity ladder**：token/edge/span DEAD → block (8.33%) → prompt (13.33%) → gen_length (20%) → H3 pass@N (86.67%) → full union 91.67%
2. **No local fix**：confidence-based revise 在任何粒度都不 work，"模型错得很自信"
3. **Write-space > diversity**：A6-only `{19, 51}` 在 H3 下仍 stuck → gen_length lever 非 diversity lever 可替代
4. **True capacity ceiling = 5 prompts** `{4, 5, 14, 41, 42}`，所有方法都救不了
5. **H3 ⊆ A-union = 19.2%**（跨轴近乎正交，但 FAIL18 内 H3 无独有 rescue）

### 哪里跟预期不符

- 期望 H1/A3 至少有弱信号（5-10%） → 实际 0。"conf = error signal" 整条路线被杀死。
- 期望 CoT 前缀稳定帮忙 → `cot_step` 砸了 12 条、`cot_plain` 砸 7 条。`answer_marker` 反而最好。
- 期望 H3 pass@N 是 capacity ceiling 的正面证据 → 实际 REJECTED ceiling 假设（反向证据：模型能力没到上限，而是被默认配置限住）。

### 教训

- **先验信号的方向经常错**：H1/A3 的 REJECTED 和 H3 的反向 REJECTED 都出乎意料
- **粒度阶梯是 paper 的骨**：比任何单实验结论都稳
- **Cross-ref script 容易有 silent bug**（P5 → P6），schema 不对就完全无声，要加断言 / 非零输出检查
