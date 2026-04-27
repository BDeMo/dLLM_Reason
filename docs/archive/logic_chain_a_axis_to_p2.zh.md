# 逻辑链：A 轴 → T6 → P2 Decode Frontier

> 语言：中文 | English: [logic_chain_a_axis_to_p2.md](logic_chain_a_axis_to_p2.md)

**目的**：把项目从 A 轴 fail-rescue 实验起一直到当前 P2 decode_ablate 的完整推理链整理成一篇文档，方便后续 T7 / Verifier / RL 的决策有清晰锚点。

**最后更新**：2026-04-27（T7 在跑，结果区域留空）

**关联文档**（详细数据 / 方法论各自有归档，本文只做串联）：
- `definitions_hard_sets.zh.md` —— 术语权威定义
- `finding_a_axis_exploration.zh.md` —— A 轴细节
- `finding_t6_training_ceiling.zh.md` —— T6 24-ckpt 消融
- `finding_p2_decode_frontier.zh.md` —— P2 decode_ablate + SC
- `runs/validation/t6_*ablate*/summary.md` —— 表格数据

---

## 0. 任务定义

**Scope**：gsm8k test split（1319 题）。
**Baseline**：LLaDA-8B-Instruct 在 canonical config（T=0 greedy, gen_length=128, block_length=32, low_confidence remasking, steps=128）下推理。
**`scope_fail`**：baseline 答错的 prompt 集合 —— v1.5 时期 60 条，v1.6.1 重定义为 331 条（canonical 重做）。
**`scope_ok`**：baseline 答对的 prompt 集合 —— 988 条。
**Rescue rate**：trained / decoding-modified 模型在 `scope_fail` 上的 pass@1。
**Retention rate**：在 `scope_ok` 上保持答对的比例。
**核心 trade-off**：`max(rescue) s.t. retention ≥ 高水准`。

---

![Capacity ladder](../figures/capacity_ladder.png)

> **图 0**：每条 ceiling 当前的实测 / 估算 rescue 率。greedy 28% → SC 38% → oracle pass@N 66% → A 轴 union 91% → A+T6+? 待解锁。注意 60-scope 与 331-scope 不同口径，取趋势看。

---

## 1. Stage 1：A 轴探索 —— 不动模型，只动推理

**问题**：base LLaDA 在 fail 集上 0%，纯推理超参/方法能救多少？

**做法**：在原 60-prompt scope 上扫各种 inference-time 干预：

| 方法 | 粒度 | 结果 |
|---|---|---|
| A1 DAG search（greedy / NAS / E2E）| 单条 DAG 边 | **0/1319+200+106 全部** —— 三证 DEAD |
| A2/H1 single-token revise | 单 token | 0/137 REJECTED |
| A3 span revise (window) | 4-token 窗 | 0/60 REJECTED |
| **A4 block layout rerank** | 整块（8-64 token）| **5/60 = 8.33%** ★ 第一个正信号 |
| **A5 prompt template** | 输入 | 8/60 = 13.33% |
| **A6 gen length** | 生成预算（64-256 token）| **12/60 = 20%** ★ A 轴最强单旋钮 |
| H3 pass@N (T>0, n=8) | 采样多样性 | **52/60 = 86.67%** ★ ★ |

**关键反转**：
- 旧 H2 假设 "block-level order 信号弱于 content"（variance ratio 应 < 0.3）—— 实际 ratio 0.754，**REJECTED**。意味着 block-level order 已经带 75% 的输出方差，A1 的 edge-level revise 是错的粒度。
- A 轴所有方法 **union** 救回 55/60 = 91.67%。
- 剩下 **5 条永远救不回**：`Ceiling-5 = {4, 5, 14, 41, 42}` —— A 轴的 capacity ceiling。

![A-axis methods](../figures/a_axis_methods.png)

> **图 1**：A 轴各方法的 rescue rate（60-prompt scope）。粒度由细到粗：单 token / span / block / template / gen-length / pass@N。
> **DEAD（红）**：edge 和 token 粒度全 0。
> **SUPPORTED（橙）**：block 粒度起开始有信号（A4=8.33%, A5=13.33%, A6=20%）。
> **Pass@N**（绿）：T>0 采样能救 86.67%，但需要 oracle 挑对的样本。
> **All-A union（深绿）**：91.67%，剩 5 条 = Ceiling-5 永远救不回。

**结论**：纯推理时干预**有上限**（91.67% on 60-scope, ~5 条 hardcoded ceiling）。pass@N 是单维度最强 lever。但 **pass@N 需要 oracle picker**（看 GT 挑对的样本），不是 deployable。

**A 轴产生的关键概念**：
- "干预粒度" 框架：edge < token < window < block < prompt < sample
- "Ceiling-M"：方法类 M 的不可救集合
- pass@N 给出 capacity ceiling 的下界

---

## 2. Stage 2 起点的问题

A 轴说"推理时能救 91% 但需要 oracle"。两条延伸：

**A. 转向训练**：能不能让模型在 greedy（无 oracle）时就把 capacity 用出来？
**B. 转向 deployable sampling**：把 oracle pass@N 兑现成 SC / BoN 等可部署 metric？

我们**先做 A**（v1.6.1 / T6），目标是验证 SFT 能否把 capacity collapse 进 mode → 之后可以再做 B（P2）。

---

## 3. Stage 2：T6 teacher-trace SFT（v1.6.1）

**Scope 重定义**：原 60 → 331（canonical eval 重跑 gsm8k test 1319 后的 fail 集）。

**做法**：
1. Qwen3-8B 在 gsm8k train（2000）上生成 reasoning trace（"teacher trace"）
2. 用 (prompt, teacher_trace) 对做 SFT 训 LLaDA-8B
3. Hyperparameter ablate：
   - **Full-SFT** epoch ∈ {0.5, 1, 2, 4} → 4 ckpt
   - **LoRA** rank ∈ {1, 2, 4, 8, 16} × epoch ∈ {0.5, 1, 2, 4} → 20 ckpt
   - 共 24 ckpt

**实施踩坑（v1.6.1 audit 6 个 bug，详见 `minors.zh.md`）**：
- 第一次跑 max_steps=2000（=12 epoch），灾难性遗忘 → fail +26.6% 但 ok -27%，net -179
- B3：Finetuner 缺 rank-0 guard，rank race 把 best.pt 写成空状态
- B6：val_loss 没 all-reduce → 各 rank 决策不一致
- ... 修完 6 bug 重跑

**修后 T6 数据（canonical T=0 pass@1）**：

| Mode | best ep | step | fail rescue | ok retain | net |
|---|---|---|---|---|---|
| Full-SFT | 2 | 336 | **28.1%** | 91.6% | **+10** |
| LoRA r=1 | 4 | 672 | 10.6% | **97.6%** | +11 |

**关键观察 1：Full vs LoRA 是 trade-off 不同的 Pareto 点**
- Full-SFT：fail 大幅救回（+93），但 ok 大幅丢（-83）。"重模型改造"
- LoRA：fail 适度救（+35），ok 几乎不丢。"轻插件"

![T6 Pareto](../figures/t6_pareto.png)

> **图 2**：T6 24 个 ckpt 的 Pareto 散点（331-scope）。
> 红方块 = Full-SFT 4 ep 点（28% fail / 91% ok 范围）。
> 圆点 = LoRA r ∈ {1,2,4,8,16}，颜色按 rank 区分，6-12% fail / 95-98% ok 范围。
> 绿区 = ok ≥ 95% 的"安全"区，LoRA 都在里面，Full-SFT 都不在。
> 任何点都没碰到 (100, 100) —— 训练侧 capacity 有上限。

**关键观察 2：T6 hardset = 166/331（50%）**

- 24 个 ckpt **没有任何一个**能救的 fail prompt = 166 条
- 即"oracle ensemble" 24 ckpt × T=0 pass@1 上限就是 49.8%
- 训练超参的边际收益已**枯竭**：再加 epoch / 加 rank 也救不回这 166 条

![T6 hardset histogram](../figures/t6_hardset_histogram.png)

> **图 3**：每条 fail prompt 被多少 ckpt 救过的分布。**红色 bar = 166 条 hardset（被 0 个 ckpt 救过）**，与所有其他 bar 加起来等量。橘色（被 1-2 个救）的 35+31 条是 "脆弱 rescue"，绿色长尾（被 ≥7 个救）48 条是"稳健 rescue"。形态是双峰：要么没人救得回，要么大家都救得回。

**关键观察 3：A-axis Ceiling-5 ∩ T6 hardset = ∅**

`Ceiling-5 = {4, 5, 14, 41, 42}`（A 轴推理时救不回）—— **全部 5 条都被 T6 某个 ckpt 救回**了：
- step_336 破 3/5
- step_84 / step_672 各破 2/5（互有重叠）
- 24 ckpt union 覆盖 5/5

**含义**：**A 轴 ceiling 与 T6 ceiling 正交**。A 轴推理时救不回的 5 条，训练能救；T6 训练救不回的 166 条，可能推理时能救。**两轴联合潜力远高于单轴**。

![Cross-axis Venn](../figures/cross_axis_venn.png)

> **图 4**：A 轴 Ceiling-5（5 条 inference 救不回）与 T6 hardset（166 条训练救不回）**完全不相交**。这是项目最重要的实证：单轴 ceiling 不是 capacity ceiling。两轴方法各自的失败模式是正交的，组合（training + decoding）必能突破任一单轴。

---

## 4. Stage 3：P2 — Decode 策略 on T6 ckpts

**问题**：A 轴和 T6 既然正交，T6 训过的模型 + A 轴推理超参联合能救多少？

**做法**：在 3 个最强 T6 ckpt（Full step_336、Full step_84、LoRA r=1 step_336）上做 decode_ablate：T ∈ {0.3, 0.7, 1.0} × N=8 × full scope (331+988)。

**结果矩阵（节选最强）**：

| Ckpt | T | pass@8 fail | **SC@8 fail** | gap | ok pass@8 |
|---|---|---|---|---|---|
| Full step_336 | 1.0 | **65.9%** ☆ | 36.6% | -29.3% | 98.7% |
| Full step_336 | 0.7 | 61.3% | **38.4%** ★ | -22.9% | 98.3% |
| Full step_84 | 1.0 | 59.8% | 31.1% | -28.7% | 99.0% |
| LoRA r=1 step_336 | 1.0 | 56.5% | 28.1% | -28.4% | 98.9% |

★ = 当前最强 deployable（SC@N pareto）
☆ = 当前 oracle ceiling

**关键观察 4：pass@N capacity ceiling = ~66%**

T6 + sampling 在 full scope 上能救 **65-66%** 的 fail（A 轴时期 30+30 测的 70-77% 是高方差小样本）。这是当前 model + decoding 联合的真实上限。

**关键观察 5：SC@N gap 异常大（25-30%）**

行业典型 oracle pass@N 与 SC@N 差 10-15%。我们 **25-30%**。意思：

- 8 个 sample 里**对的那个**经常孤立（少数派 1-2 票）
- "错的多数派" 有**系统性偏好** —— 模型在 fail prompts 上**没收敛到正解**
- majority vote 投了错的，正解被淹没

**关键观察 6：SC vs greedy 提升只 ~10%**

```
greedy T=0 pass@1 (Full-SFT step_336):  fail 28.1%, ok 91.6%
SC@8       T=0.7 (Full-SFT step_336):   fail 38.4%, ok 94.7%
                                          ────       ────
                                         +10.3%    +3.1%
```

SC 单独能 deploy，**但远没到 65% capacity 上限**。剩 ~28% rescue 是 oracle 才拿得到的。

![pass@N vs SC@N](../figures/passN_vs_SC.png)

> **图 5**：3 ckpt × 3 温度 × pass@8 vs SC@8 对照。蓝条 = oracle pass@8（capacity 上限），红条 = SC@8（majority vote，可部署）。中间灰色箭头标注 gap。**所有 cell gap 都在 22-30%**，远超行业典型 10-15%。这意味着模型在 fail prompts 上**没收敛到正解**：8 个 sample 里偶尔产 1-2 个对的，剩 6-7 个有系统性偏好的错答案，majority 投错的。

**关键观察 7：T 越高 pass@N 越高，但 SC 不一定**

```
fail pass@8 vs T (step_336):    0.3 → 0.7 → 1.0  单调升 (50.2 → 61.3 → 65.9)
fail SC@8   vs T (step_336):    0.3 → 0.7 → 1.0  非单调 (34.1 → 38.4 → 36.6)
```

最佳采样温度 T=1.0，最佳 SC 温度 T=0.7。**diversity（pass）和 reliability（SC）的最优 T 不同**。

---

## 5. 当前的逻辑结论（决定下一步）

### 三组 ceiling 对比

```
A 轴 ceiling                 T6 ceiling                  联合 ceiling
inference-only @ T=0 pass@1  training-only @ T=0 pass@1  A + T6 + sampling
55/60 (91.67%) on old scope  165/331 (49.8%) on new      ?? 还在测
                             scope（166 不可救）
```

### 当前 deployable 实际数

```
greedy (T6 best @ T=0):              28.1% fail, 91.6% ok
SC@8   (T6 best @ T=0.7):            38.4% fail, 94.7% ok
oracle (T6 best @ T=1.0 pass@8):     65.9% fail, 98.7% ok    ← 不可部署
真 capacity (24 ckpt union):         50.2% fail (= 165/331)
```

### Gap 分析

| Gap | 数值 | 解读 |
|---|---|---|
| `pass@N − greedy` | **+38%** | 模型有 capacity，greedy 取不出来 |
| `pass@N − SC@N` | **+27%** | majority vote 漏掉 oracle 能挑的 |
| `SC@N − greedy` | **+10%** | sampling+vote 实际能 deploy 的提升 |
| `T6_hardset` | 50% | 训练侧不可达（需别的 axis）|
| `Ceiling-5 / 60` | 8.3% | A 轴推理不可达（被 T6 cover 了）|

### 决策树

```
当前 deployable = SC@8 = 38%
真正的 capacity 上限 = pass@N = 66%
必须解决：把 27% 的 pass-SC gap 兑现到 deployable

候选路径（按预期 ROI）:

1. T7 self-distill ★（in-progress）
   把 pass@N 的正样本做 SFT → 模型直接 greedy 输出对答案
   期望：T=0 pass@1 从 28% → 45-55%
   状态：v1 失败（pick=shortest + over-train），v2 跑中

2. ORM / BoN
   训对/错 verifier head，N 个 sample 选最高分
   期望：把 38% SC → 50-60%
   状态：infra ready (correction_train)，未启动

3. PRM-RL
   step-level reward + GRPO
   期望：50-65%，但训练不稳
   状态：infra 大部分 ready (rl_train.py)，未启动

4. 数据 / 训练侧扩展
   - 更大 SFT 数据集（gsm8k 之外）
   - 多样化 teacher（Qwen + DeepSeek + Llama）
   - 状态：完全未做
```

---

## 6. T7 实验记录区（待回填）

> ★ T7 v1 (2026-04-26) **FAILED**：pick=shortest 选了 truncated 样本 + max_steps=1500 = 6 epoch over-training。
>
> Canonical T=0：fail 27.2%（vs T6 28.1%），ok 88.4%（vs T6 91.6%），net -35。
>
> Decode_ablate full scope（2026-04-27 跑完）：T7 v1 vs T6 best 在所有维度都差：
> | metric | T6 step_336 | T7 v1 | Δ |
> |---|---|---|---|
> | greedy fail | 28.1% | 27.2% | -0.9% |
> | fail pass@8 (oracle, T=1.0) | 65.9% | 59.2% | **-6.7%** |
> | fail SC@8 (best) | 38.4% | 36.3% | -2.1% |
> | ok pass@8 | 98.7% | 98.7% | 0 |
> | ok SC@8 | 95.6% | 93.0% | -2.6% |
>
> 关键观察：**capacity 上限本身被压低 7%**（pass@8 从 66 → 59）。意味着 trajectory-level SFT 用垃圾数据**主动伤害了 sampling diversity**，不只是没 collapse 进 mode。
>
> ★ T7 v2 (2026-04-27 计划)：repick existing per_prompt with `pick=first` + max_steps=480（2 epoch）。复用 v1 的 1918-prompt cover_rate=95.9% 的 candidates。
>
> ★ T7 v2 数字（待跑完回填）：
> - canonical T=0 pass@1: ?
> - canonical ok retention: ?
> - decode_ablate pass@8 ceiling: ?
> - SC@8 best: ?
> - 与 T6 net delta: ?
>
> ★ 决定下一步的判据：
> - if T7 fail ≥ 45% → success path，T7 当 prod baseline
> - if 32% < T7 fail < 45% → partial，扫 epoch / 换 gen_ckpt
> - if T7 fail < 32% → self-distill 死，切 ORM / RL 路线

---

## 7. Cross-stage 学到的 meta 教训

1. **Ceiling 总是相对方法类 M 定义的** —— 不要混用 "hardset"、"ceiling"。每加一类方法 M，ceiling 缩小。
2. **正交两轴可以复合突破** —— A-axis Ceiling-5 ∩ T6_hardset = ∅ 是关键证据。任一单轴的 ceiling 不是绝对 capacity ceiling。
3. **小样本（30+30 subset）会骗你** —— pass@N 在 30+30 上 70-77%，full scope 落到 65%。CI ±18% 太松。
4. **训练超参枯竭比想象快** —— 24 个 ckpt 的 union 也只到 50%，加 rank/epoch 已经没空间，必须换轴。
5. **Audit 永远值得** —— v1.6.1 第一次跑 net=-179，audit 修 6 个 bug 后 +10。**没 audit 的 pipeline 不要相信数字**。
6. **pass@N vs SC@N 的 gap 是 hidden signal** —— gap 大说明模型在 fail set 上**没收敛**，不仅是"采样不够多"。这是 training-side 还有空间的暗示（如 T7、RL）。

---

## 8. 图表索引

所有图均由 `scripts/make_logic_chain_figures.py` 生成（idempotent，新数据落盘后重跑即可刷新）。

| 编号 | 文件 | 内容 |
|---|---|---|
| 0 | `docs/figures/capacity_ladder.png` | 能力天花板梯度图（baseline → SC → pass@N → A 轴 union） |
| 1 | `docs/figures/a_axis_methods.png` | A 轴 6 方法 + union rescue rate bar chart（60-scope） |
| 2 | `docs/figures/t6_pareto.png` | T6 24 ckpt Pareto 散点（fail vs ok, 331-scope） |
| 3 | `docs/figures/t6_hardset_histogram.png` | rescue-count 分布直方（166 hardset 双峰）|
| 4 | `docs/figures/cross_axis_venn.png` | Ceiling-5 ∩ T6_hardset = ∅ Venn |
| 5 | `docs/figures/passN_vs_SC.png` | 3 ckpt × 3 T × pass@8 vs SC@8 对照 |

## 9. 文件 / 数据导引

| 概念 | 主文档 | 主数据 |
|---|---|---|
| 术语 | `definitions_hard_sets.zh.md` | — |
| A 轴 | `finding_a_axis_exploration.zh.md` | `runs/validation/{a4_*,a5_*,a6_*,h3_*}/` |
| T6 训练 | `finding_t6_training_ceiling.zh.md` | `runs/validation/t6_ablate/`、`t6_lora_ablate/`、`t6_hardset/` |
| P2 decode | `finding_p2_decode_frontier.zh.md` | `runs/validation/t6_decode_ablate/` |
| Bug 日志 | `issues/minors.zh.md` | — |
| T7（in progress）| 本文 §6 | `runs/validation/t7_gen_*/`、`t7_eval_*/` |
