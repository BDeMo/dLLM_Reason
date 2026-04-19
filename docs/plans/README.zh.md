# 项目 Plans 目录

> 语言：中文  |  English: *(TODO — EN mirror)*

**创建日期**：2026-04-19
**目的**：所有阶段性计划、设计决策、事后分析（postmortem）、想法的**时间序**档案。每次有新 plan、新阶段、新想法，新建一个**日期前缀**的文件放进来；小的更新 append 到现有文件的 Updates 段。

---

## 为什么有这个目录

跟 `docs/archive/` 的区别：
- `archive/` = **研究结果/findings**（事实 + 数字 + verdict）—— 永远 true 的东西
- `plans/` = **过程/决策/想法**（为啥选这个方向，当时权衡什么，后来走歪了哪里）—— 有时间语义的东西

`archive/` 是"**论文附录**"，`plans/` 是"**研究日志**"。每当有人（包括未来的自己）问"为什么当初这么做"，答案在 `plans/`。

---

## 维护规则

### 什么时候 append 新文件

| 触发 | 操作 |
|---|---|
| 新实验阶段开始 | 新文件 `YYYY-MM-DD_phase_name.zh.md`，状态 `active` |
| 大的设计决策（影响超过 1 天的工作量） | 新文件 |
| 实验挂了 / 跑不出来 → postmortem | 新文件 `YYYY-MM-DD_xxx_postmortem.zh.md` |
| paper framing 调整 | 新文件 |
| 小想法 / 调试笔记 / 中期调整 | **append 到当前 active plan 的 Updates 段**，带日期 |

### 命名规则

```
YYYY-MM-DD_<topic-snake-case>.zh.md
```

例：`2026-04-19_ss_run_postmortem.zh.md`

文件按文件名字母序排 = 时间序（YYYY-MM-DD 前缀保证）。

### 文件头模板

每个 plan 文件必须有如下 header：

```markdown
# <Plan 标题>

**日期**：YYYY-MM-DD（首版）/ YYYY-MM-DD（最近更新）
**状态**：active / superseded / abandoned / done
**前置上下文**：触发这个 plan 的上一个事件（指向某个 archive doc 或别的 plan）
**继任**：（如果被 superseded，指向下一个 plan）

## 目标
<1-3 句>

## 决策 / 方法
<主体>

## 预期输出
<要产出什么 artifact / claim>

## Updates
- **YYYY-MM-DD**: <更新内容，带时间戳>
- **YYYY-MM-DD**: <...>

## Retrospective（done 或 abandoned 时填）
- 实际做了什么
- 哪里跟预期不符
- 教训
```

### 状态流转

```
active ─┬─> done                    # 按计划完成
        ├─> superseded              # 被后续 plan 取代（正常替换）
        └─> abandoned               # 放弃了（记下为什么）
```

---

## 时间线（全量索引）

按日期倒序（最新的在上）。每次加文件**记得更新这张表**。

| 日期 | 文件 | 状态 | 摘要 |
|---|---|---|---|
| 2026-04-19 | [`2026-04-19_related_work_review.zh.md`](2026-04-19_related_work_review.zh.md) | active | 15 篇精读：LogicDiff / DAWN / Where-to-Unmask / ReasonFlux / ... + paper positioning |
| 2026-04-19 | [`2026-04-19_replan_next_phase.zh.md`](2026-04-19_replan_next_phase.zh.md) | **active** ⭐ | 当前 active。SS 搁浅后重新规划 |
| 2026-04-19 | [`2026-04-19_ss_run_postmortem.zh.md`](2026-04-19_ss_run_postmortem.zh.md) | done | SS 跑 3 天 13.8% 完成度，慢 11× 未定因。止损 |
| 2026-04-16 | [`2026-04-16_paper_framing.zh.md`](2026-04-16_paper_framing.zh.md) | active | Paper 定位 "canvas-constrained reasoning" |
| 2026-04-16 | [`2026-04-16_distillation_decisions.zh.md`](2026-04-16_distillation_decisions.zh.md) | active | Distillation 4 决策：cheapest / key=value / 裸 prompt / abstain |
| 2026-04-16 | [`2026-04-16_strategy_search_design.zh.md`](2026-04-16_strategy_search_design.zh.md) | superseded | SS 5D 搜索设计。被 2026-04-19_replan 取代 |
| 2026-04-15 | [`2026-04-15_a_axis_discovery_phase.zh.md`](2026-04-15_a_axis_discovery_phase.zh.md) | done | A1-A6 + H3 扫完，全方法 union = 91.67% |

---

## 当前 active plan

**👉 [`2026-04-19_replan_next_phase.zh.md`](2026-04-19_replan_next_phase.zh.md)** —— 最新决策

**👉 [`PROJECT_OVERVIEW.zh.md`](PROJECT_OVERVIEW.zh.md)** —— top-level 项目总览（常驻，不带日期）

---

## 相关入口

- [`docs/archive/`](../archive/) —— 研究事实 / findings / 消融索引
- [`docs/archive/ablation_index.zh.md`](../archive/ablation_index.zh.md) —— 所有实验代号/verdict/数字的总索引（含 Setting & Definitions）
- [`docs/archive/hypotheses.zh.md`](../archive/hypotheses.zh.md) —— H 轴假设登记簿
