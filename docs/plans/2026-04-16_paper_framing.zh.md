# Plan: Paper Framing —— "Canvas-Constrained Reasoning"

**日期**：2026-04-16
**状态**：active（定位已敲定，具体 title 待 SS 数据落地后选）
**前置上下文**：用户问"论文叫啥名" + "归类 constrained GenAI 的 constraint 是啥"
**继任**：—

## 目标

给 paper 一个**可辨识的 framing**：能在 constrained GenAI / reasoning-in-LM / diffusion-LM 三个 community 定位清楚。

## 决策

### Framing：**"Canvas-Constrained Reasoning in Discrete Diffusion Language Models"**

AR LM 和 diffusion LM 的根本区别（我们的 paper 立论点）：
- AR LM：输出长度是 **emergent** 的（EOS token 决定），canvas 是"无限长卷轴"
- Diffusion LM：输出长度是 **预先指定的硬约束**（`gen_length` 采样前给定），canvas 是"固定画布"

→ **Core constraint = fixed finite write-space，pre-committed before inference**。

### Constraint 维度（= 我们扫的 axes）

| Axis | Constrained-GenAI 术语 |
|---|---|
| A6 gen_length | **Budget / canvas-size constraint** |
| A4 block layout | **Schedule / denoise-path constraint** |
| A5 prompt template | **Framing / canvas-prefix conditioning** |
| Template_position（SS 新） | **Positional scaffolding / inpainting constraint**（diffusion LM 独有）|
| H3 pass@N | Diversity **relaxation**（不是 constraint）|
| A1 DAG edge | **Order constraint** at edge level（DEAD）|
| A2/A3 conf revise | **Dynamic commit constraint**（runtime, DEAD）|

### Paper 可卖的两条 story

**Story A —— "Structural vs Dynamic constraints"**
- 结构性/预设 constraints（canvas size/layout/framing）**有信号**
- 动态/运行时 constraints（conf-based revise）**全死**
- 一句话：*"In discrete diffusion LM reasoning, only pre-committed structural constraints rescue errors; runtime confidence-based constraints carry no signal."*

**Story B —— "Inpainting-as-reasoning-scaffold"**（最独特）
- Diffusion LM 天生支持 inpainting（canvas 任意位置可填）
- AR LM 做不到
- Paper 核心 claim：*"Canvas-level constraints unique to diffusion LMs (position-aware scaffolding, block schedule, gen-length budget) open a rescue-space an AR LM cannot access."*

### Title 候选（ranked）

1. ⭐ **"No Local Fix: A Granularity Ladder for Rescuing Reasoning Errors in Discrete Diffusion Language Models"** —— strong negative result + unique concept
2. "Canvas-Constrained Reasoning: What Rescues Discrete Diffusion LMs on Math Word Problems"
3. "Write-Space Over Diversity: What Actually Rescues Diffusion LM Reasoning"
4. "The Canvas Is the Constraint: Inference-Time Rescue in Discrete Diffusion LM Reasoning"

### 跟主流 constrained GenAI 的 differentiator

主流方向（Outlines / Guidance / grammar-constrained decoding / JSON mode）约束的是 **"输出必须匹配某个正则/语法/schema"** = **output-space constraint**。

我们约束的是 **"canvas 本身的形状、大小、位置结构"** = **generation-substrate constraint**。

- Output-space constraint 回答："输出长什么样"
- Canvas constraint 回答："模型在什么形状的纸上作答"

两者**正交**，这是 related work 的主要卡位。

## 预期输出

- Paper 提交前的 section 列表：
  - **§1 Intro**：canvas constraint 的 motivation
  - **§2 Related work**：跟 constrained decoding 的正交性
  - **§3 Methods**：granularity ladder + 5D search space（含 template_position 是新维度）
  - **§4 Results**：A 轴 + H3 + SS oracle rate + FAIL18 逐条
  - **§4.3 Write-space over diversity**（A6-only {19, 51} + idx=48 的叙事）
  - **§5 Discussion**：structural vs dynamic constraints 的 story
  - **§6 Limitations & future work**：learned policy head (H4), verifier (B4), tool-use (B3)

## Updates

- **2026-04-16**: framing 敲定，subtitle 暂定 "granularity ladder"
- **2026-04-19**: SS 搁浅，**core claim 的实验证据目前只有 A 轴 + H3**（不含 template_position novelty）。paper 可能要 pivot 到只讲 structural vs dynamic + granularity，暂时不讲 canvas inpainting 独家贡献 —— 除非 replan 里能快速出 template_position 数据

## Retrospective（pending）

Canvas-constrained 的核心 claim 强烈依赖 SS 里 template_position 的**独家 rescue**存在。如果 SS 最后证明 `inpaint_novel_set` 为空，paper 要：
- 降格 canvas framing 成"gen_length 是 canvas budget 的 proxy"
- 或者换成纯"granularity ladder" story（A 轴已足够）
- 放弃 inpainting-as-scaffold 这条独特卡位

需要有意识地在 SS replan 里优先保证 template_position 覆盖。
