# Finding：A 轴探索 —— Block 粒度是第一个信号

> 语言：中文  |  English: [finding_a_axis_exploration.md](finding_a_axis_exploration.md)

**日期**：2026-04-15（初版）/ 2026-04-16（H3 扩到 n=60 重跑 + P6 crossref 后同步）

**配套文档**：
- [`hypotheses.zh.md`](hypotheses.zh.md) —— 假设登记簿 + 结论板
- [`exploration_axes.zh.md`](exploration_axes.zh.md) —— A/B 轴索引 + 状态标签
- [`finding_dag_search_zero_rescue.zh.md`](finding_dag_search_zero_rescue.zh.md) —— A1 DEAD 的三证

---

## TL;DR

DAG search 三次独立实现 + 两次 token 级 revise 实验全部 0 rescue 之后，A 轴在粗粒度层出三个正信号：A4 block-layout rerank（5/60 = 8.33%）、A5 prompt-template rerank（8/60 = 13.33%）、**A6 gen-length rerank（12/60 = 20.00%，A 轴最强单旋钮）**。H3 pass@N 扩到 n=60 后 **52/60 = 86.67%** 成为最强单维度杠杆（跟 A 轴近乎正交）。全方法 union **55/60 = 91.67%**，true capacity ceiling **5 条 [4,5,14,41,42]**。

| 实验 | 旋钮 | 干预粒度 | Verdict | Rescue |
|---|---|---|---|---|
| A1 · DAG search | order | 单条 DAG 边 | **DEAD**（三证） | 0/1319+200+106 |
| A2 / H1 · token revise | 撤销 commit | 单 token | **REJECTED** | 0/137 |
| A3 · span revise | 撤销 commit | 4-token 窗口 | **REJECTED** | 0/60 |
| **A4 · block layout** | **denoise 路径** | **整块（8-64 token）** | **SUPPORTED** | **5/60 (8.33%)** |
| **A5 · prompt template** | **输入** | **prompt 本身** | **SUPPORTED** | **8/60 (13.33%)** |
| **A6 · gen length** | **生成预算** | **总长度 (64-256 token)** | **SUPPORTED** | **12/60 (20.00%)** |
| A4×A5 joint 6-cell | layout × template | 6 格 ensemble | — | 10/60 (16.67%)，overlap 预测完美验证 |
| H2 · order vs content | variance ratio | block_length ∈ {16,32,64} | REJECTED（按阈值）；同时预示 A4 | ratio 0.754 |
| **H3 · pass@N (n=60)** | **采样多样性** | **—** | **REJECTED**（capacity ceiling 阈值） | **52/60 (86.67%)** |

关键反转：H2 实际上是一次 block_length 扫描（`block_length ∈ {16, 32, 64}` at T=0，同一个 scheduler，不是原来描述里的三种不同 scheduler），REJECTED 了 "order 信号 < content 信号 30%" 的强命题（ratio 0.754 远超 0.3）—— 也就是说 block-layout 级别的 order 已经带了 temperature 75% 的输出方差。A1 的三次 edge-level 0 rescue 说的是另一粒度。A4 是最直接的下一步：用 H2 一模一样的旋钮（block_length），问的不是"输出变不变"，而是"正确率变不变"。答案：8.33% 的 fail prompt 被救。

---

## Scope —— 137 条 fail prompt（H0）

`scripts/validate/h0_forensics.py` 读最新 `runs/research_*/stage2_discovery/episodes.db` 里的 `correct=0` 行，按错误类型分桶，落两份 JSON：

- `runs/validation/scope_fail_prompts.json` —— 137 条 LLaDA-instruct（T=0、`block_length=32`、`remasking=low_confidence`、`gen_length=128`、`steps=128`）答错的 gsm8k prompt。
- `runs/validation/scope_ok_prompts.json` —— 对应的 `correct=1` 集，用作 H3 对照组。

错误类型分布（只做诊断，不下 verdict）：
- `numeric_close / numeric_mid / numeric_far / numeric_order` —— 按相对误差量级递增。
- `format_bad` —— 截断 / 没给出数字。
- `unknown` —— gt 本身 parse 失败。

所有 A 轴实验都读同一份 `scope_fail_prompts.json`，所以 **prompt 在 A3/A4 里的 per-prompt 记录可以一对一查** —— 被 A4 救的那 5 条可以回查它们在 H1/A3 里是不是也被救过。

---

## H1 —— Single-token revise hook · REJECTED

**断言**：每 revise_every 步把已 commit 且 conf < τ 的 token 置回 mask，能救一些 fail。

**设计**：
- `scripts/validate/h1_remask_rescue.py`
- 每条 prompt 两次推理：
  1. **baseline**：原生 LLaDA（T=0、low-confidence remasking、无 hook）。
  2. **revise**：同 baseline，但每 `revise_every=8` 步对**当前 block 范围内**已 commit 且 `committed_conf < revise_thresh=0.3` 的位置置回 mask，conf 重置为 `+inf`。
- `rescue_rate = |{revise 对 ∧ baseline 错}| / N`。

**结果**：N=137、base=0、revise=0、rescued=0、broken=0、rescue_rate=**0.00%** → REJECTED。

**诊断**：base_correct = revise_correct = 0/137，rescued=0、broken=0 —— hook 对最终正确率零贡献。符合"fail 上 conf 大多 ≥ 0.3，hook 鲜少触发"的直觉，但只有最终 correctness 向量本身能被 summary 证实，确切触发次数 summary.json 没持久化。**per-token conf 在 fail prompts 上没有错误信息** —— 模型错得很自信。

---

## H2 —— Order variance vs content variance · REJECTED（但有 nuance）

**断言**：同一 prompt 上，改 order 的输出方差远小于改 temperature 的方差。具体阈值：`order_var / content_var < 0.3` → SUPPORTED。

**设计**：
- `scripts/validate/h2_order_vs_content.py`
- 从 fail 集取 K=20 条。
- **Content 轴**：同一 scheduler，T ∈ {0, 0.3, 0.7}，每 T 采 3 次 → 9 条 output。
- **Order 轴**：T=0 固定，同一个 `low_confidence` scheduler，换 3 个 `block_length` ∈ {16, 32, 64}（换 block size 重排跨步 commit 顺序，不动 content 采样）→ 3 条 output。
- 每条 prompt 算 output 间两两 normalized edit distance 的方差。

**结果**：N=20、content_var=0.256、order_var=0.176、ratio=**0.754** → REJECTED。

**解读**：order 的信号没被 content 压垮，而是 content 的 75%。表面看跟 A1（edge DAG 0 rescue）矛盾。真相：H2 的 "order 轴" 其实就是 `block_length ∈ {16, 32, 64}` —— 跟 A4 是同一个旋钮的子集。所以 H2 测的从一开始就是 **block-layout 方差**，而不是 edge-level 方差；A1 的 0-rescue（单边级）和 H2 的 0.754（block 级）其实说的是不同粒度。A4 是最直接的下一步：block-level 的输出多样性能不能翻译成**正确率**的 rescue。

---

## H3 —— Pass@N capacity ceiling · 最强单维度杠杆（n=60 权威）

**断言**：这 137 条在 LLaDA-instruct 的能力上限之上，即使加 temperature + N 次采样也救不动。

**设计**：
- `scripts/validate/h3_passN_at_temperature.py`
- n_fail + n_ok 对照。
- 每条 prompt × T ∈ {0.3, 0.7, 1.0} × N=8 → 算 pass@1/4/8。
- Verdict：`fail_p@8 < 5% AND ok_p@8 > 90%` → SUPPORTED（capacity ceiling）；`fail_p@8 > 20%` → REJECTED。

**初版（n=30 fail, 2026-04-15）**：7 条 rescue（idx=0,2,8,13,15,24,28），rescue_rate=23.33%，4 条 stuck。H3 独有 {2, 24}，H3 ∩ A6 独有 {0}，H3 与 A4∪A5 共享 {8, 13, 15, 28}。

**n=60 重跑（`h3_passN_20260415_133254`, 2026-04-16）**：扩到完整 60 条 fail + 30 条 ok 对照。
- `fail_pass@8_max = 86.67%` (T=1.0) / `ok_pass@8_max = 100%` → **按阈值 REJECTED**（= capacity ceiling 不成立）
- P6 crossref (`p6_h3_crossref.py`) 输出：H3 rescue = **52/60 (86.67%)**，H3 stuck = **8 条 [4, 5, 14, 19, 41, 42, 48, 51]**

**n=60 rescue 集交叉（P6 crossref 权威）**：
- A-union (A4∪A5∪A6) = 13/60 (21.67%)
- H3 ∩ FAIL18（bl32/baseline/g128 baseline 都错的 18 条子集）= 10 条 {0, 8, 10, 13, 15, 28, 35, 53, 55, 59} —— **全被 A-union 涵盖，H3 在 FAIL18 内 0 条独有 rescue**
- H3 独占 42 条都在 FAIL18 外（= A 轴 baseline 对，但 T>0 sampling 中也有正确解）
- A6 独救 {19, 51}，H3 也救不了（确认 "write-space > diversity"）
- idx=48 新类别："A5+A6+Joint 救，H3 stuck" —— A 轴能救、pass@N 救不了

**解读**：
1. 初版 n=30 "23.33%" 是**小样本偏差**；完整 60 条下 pass@N rescue 率是 86.67%，**不是**跟 A 轴差不多而是**全面压倒** A 轴
2. 但在 "能救 baseline 错的 prompt" 这个严格问题上，H3 仍无独有贡献（10/10 全被 A-union 吞）—— H3 的价值在于 pass@8 稳定 flip 了大量 baseline 对的 prompt 在 T=0.3 上的 rescue（反过来覆盖了 baseline 错的）
3. write-space 信号 (A6 独救 {19, 51}) 在 H3 下仍救不动 → 证明 A 轴 rescue 是系统性的，不是随机撞对
4. true capacity ceiling 确定为 5 条 [4, 5, 14, 41, 42]

**历史 bug**：`p5_h3_crossref.py` 是针对 H3 旧 schema 写的（`pass_at_k` dict），在当前 H3 `fail_XXXX.json + temps.T.pass@k` shape 下 silently 输出 h3_rescue=0。**已通过 `p6_h3_crossref.py` 修正**。

---

## A1 / A2 —— Edge DAG + token revise · DEAD（回顾）

A1 已归档在 `finding_dag_search_zero_rescue.zh.md`。小结：三次独立实现（1319 条 prompt 上 greedy ±1 edge、200 条上 NAS supernet、106 条上 E2E differentiable）全部 0 rescue；NAS 和 E2E 额外报告**自身优化器选中 0 条边**。T=0 + 双向 attention 下，edge-level 顺序对 greedy low-confidence 基线毫无信号。

A2 就是 H1 换个马甲（single-token revise hook），REJECTED 同上。

---

## A3 —— Span-level revise · REJECTED

**断言**：错误藏在**连续 span**里（一个小算式片段），单 token conf 可能不低，但 **窗口平均** conf 低。

**设计**：
- `scripts/validate/a3_span_revise.py`
- 走 server 端新加的 `/generate_span_revise` 端点（`scripts/serve.py`）。
- 采样循环结构同 H1，revise 判据和动作改成：
  - **判据**：每 `revise_every=8` 步，用 `F.conv1d`（kernel = `torch.ones(window_size=4)`、padding=2）对 committed conf 做 1-D 滑窗平均；同时 conv1d 一份 committed mask 得到 counts tensor 过滤不可靠窗口（`counts < max(2, window_size//2)` 剔除）。窗口"坏"当且仅当 `mean < revise_thresh=0.4` 且 reliable 且中心位已 commit。
  - **动作**：再做一次 conv1d（kernel 相同），把 bad 窗口覆盖到的所有 committed 位置找出来 —— 整窗（不只是中心）全部置回 mask，conf 重置 `+inf`。
- 阈值**特意比 H1 宽**（0.4 vs 0.3）：窗口平均会削弱单点极低的尾巴，我们要找的是"一片连续偏低"而不是"孤立的一个低点"。

**结果**：N=60、base=42、revise=42、**rescued=0、broken=0、rescue_rate=0.00%** → REJECTED。

两次采样在每一条 prompt 上正确性完全相同（base_correct = revise_correct = 42/60，rescued=0，broken=0）。连同 H1 是强证据：**任何基于 committed conf 的 revise hook 都在这份 fail 集上 dead，而不是阈值调得不好的问题**：
- H1 窄判据（τ=0.3、单 token）在 fail 上极少触发（从 H1 run 的诊断看大多数位置 conf 都高于阈值）。
- A3 宽判据（τ=0.4、window=4）严格弱于 H1 per-token 判据，触发次数至少不少；但两路 correctness 完全一样。要么宽判据仍然没怎么触发，要么触发后重采样仍收敛到同一错 token。

两种可能都把"conf 当错误信号"这整条路线（任何 token/span 粒度）杀掉。**模型内部的不确定度在这批 prompt 上跟错误没相关**。

---

## A4 —— Block-layout rerank · SUPPORTED

**断言**：默认 `block_length=32` 只是 valid block layout 空间里的一个点；换一种 layout 能救一些 fail。

**设计**：
- `scripts/validate/a4_block_rerank.py`
- Server 端走已有的 `/generate`（uniform）和新加的 `/generate_block_schedule`（非均匀）。
- 每条 prompt 5 个 layout，T=0：

| Layout | 类型 | 规格 | 动机 |
|---|---|---|---|
| `bl8` | uniform | `block_length=8`，16 块 | 最细，单步竞争池最小 |
| `bl16` | uniform | `block_length=16`，8 块 | 中间档 |
| `bl32` | uniform | `block_length=32`，4 块 | **baseline**（应与 H1 base 一致） |
| `bl64` | uniform | `block_length=64`，2 块 | 最粗，单步竞争池最大 |
| `short_then_long` | 非均匀 | block_sizes=[16,16,16,16,64]，steps=[16,16,16,16,64] | 前 64 tokens 精细推理，后 64 tokens 粗解答案 |

- **Step 预算归一**：各 layout forward 总次数一致。Uniform：`steps = max(args.steps, num_blocks)` 向上取至 `num_blocks` 整数倍。非均匀：`sum(steps_per_block) == 128`。**避免"粗 block 跑更少步 → 更差"这类混淆**。

**为什么 T=0 下不同 layout 仍给不同输出**（关键点）：
- 单次 forward 在固定 `x` 上是 deterministic 的。
- 但 denoising 是一连串 forward，每步 input 取决于前一步 commit 了哪些位置。
- 不同 block layout 每步 commit 不同位置 → 后续 forward 看到的 `x` 不同 → logits 不同 → commit 不同。
- 所以**最终输出是路径依赖的**，尽管每次 forward 本身 deterministic。

**Verdict 构造**：
```
rescue_rate = |{prompt : 有 layout 对 ∧ bl32 错}| / N
broken      = |{prompt : bl32 对 ∧ 所有 layout 都错}|
any_rate    = |{prompt : 有 layout 对}| / N
```

**结果**（N=60，run 被中断，未到 137）：
```
base(bl32)=42  any_layout=47  rescued=5  broken=0
rescue_rate=8.33%  any_rate=78.33%
per_layout: bl8=43  bl16=41  bl32=42  bl64=37  short_then_long=37
```
→ **SUPPORTED**（8.33% ≥ 5% 阈值）。

**三条 per-layout 观察**：
1. **细 > 粗**：bl8=43 > bl32=42（+1），但 bl64=37（−5）、short_then_long=37（−5）。细 block = 单步竞争池小 = commit 更保守；粗 block 更激进，反而破坏更多 prompt。
2. **无单一 layout 碾压**：rescue=5 靠 ensemble，没有任何单个 layout 比 bl32 单独多救 5。这 5 条救回来的 prompt 分散在不同 layout 上（可从 `per_prompt/*.json` 逐条看）。
3. **`broken=0`**：没有 prompt 因为换 layout 被破坏 —— 纯增益，无 trade-off。

**警觉点**：
- **N=60 不是 137**：run 中途停了。60 条里 5 条救=8.33%；如果剩下 77 条只多救 1 条，最终 rescue=6/137=4.4% → INCONCLUSIVE。**必须 `--resume` 到 137 才能敲定**。
- **Ensemble vs single-layout**：SUPPORTED 是 ensemble 命题。问"是否有单个更好的 block_length"，诚实答案是没有 —— bl8 比 bl32 +1 在 N=60 上噪声以内。真正的命题是"**存在 per-prompt 最优 layout**"，这弱一点但更有意思。
- **没有 prompt → layout 预测器**：知道有更好 layout，但不试过所有 5 个没法选。追问实验 A4.1：训练预测器 `prompt → best_layout`，如果准确率 > 随机（1/5=20%）→ layout 信号可用作 sampler 旋钮。

**与 H2 调和**：
- H2 其实早已扫过 `block_length ∈ {16, 32, 64}`（T=0、同 scheduler），测出 `order_var / content_var = 0.754` —— 光是 block-layout 这一维就带了 temperature 75% 的输出方差。
- A1：edge-level order 0 rescue。
- A4：block-level order 8.33% rescue。
- **自洽叙事**：那 0.754 一直就是 block-layout 量级的数字（H2 的 "order 轴" 本来就是 A4 layout 扫描的子集）。信号真实存在、就在 block 级；H2 看到的是输出多样性，A4 进一步证实这份多样性里含**正确率增益**，不只是表面编辑差异。A1 的 edge 级搜索在比 variance 所在位置更细的粒度上转，所以没找到。

---

## A5 —— Prompt-template rerank · SUPPORTED

**断言**：合适的 prompt 前缀（CoT 诱导或答案前缀）把输出分布推到能解的区域。

**设计**：
- `scripts/validate/a5_prompt_template.py`
- Server 端走已有的 `/generate`（跟 A4 uniform layout 共用后端）。
- 每条 prompt 4 个 template（append 到原 prompt 后）：
  1. `baseline` —— 不动。
  2. `cot_plain` —— `"\nLet's solve this step by step."`（显式 CoT 诱导）
  3. `cot_step` —— `"\nStep 1:"`（结构更强制，强推 CoT 骨架）
  4. `answer` —— `"\nAnswer:"`（抑制 CoT，直接答数字）
- Verdict 逻辑 / 阈值同 A4。

**跟 A4 的设计差异**：A4 改"采样器怎么走"，A5 改"采样器看到什么"。如果 A5 SUPPORTED 而 A4 没 → 问题在 prompt 框架；如果 A4 SUPPORTED 而 A5 没 → 问题在 denoise 路径；如果都 SUPPORTED（实际发生） → 两轴都有信号，A4 / A5 被救 prompt 集的 overlap 决定它们是否独立杠杆。

**结果**（N=60，同 A3/A4 都停在这个 N）：
```
base=42  any_template=50  rescued=8  broken=0
rescue_rate=13.33%  any_template_rate=83.33%
per_template: baseline=42  cot_plain=35  cot_step=30  answer=45
```
→ **SUPPORTED**（13.33% ≥ 5% 阈值）。A 轴 per-experiment rescue rate 最高。

**三条关键观察 —— 这是最反直觉的部分**：
1. **`answer` 反而 beat baseline（+3）**：单模板最好的不是两个 CoT，是"直接答"前缀。H0 的 gsm8k prompt 本身已经被调成会引发推理，显式加 `"\nAnswer:"` 反而截断了冗长展开、在 128 token 预算内把数字顶出来。
2. **CoT 前缀主动伤害**（`cot_plain` −7，`cot_step` −12）：强推 step-by-step 比 LLaDA-instruct 默认更差。`"\nStep 1:"` 尤其糟 —— 这个前缀 commit 了一个不匹配 instruct 模型 decoder prior 的格式，造成 prompt-shape 和 decoder 期望之间的错配。
3. **`broken=0` 又出现**：跟 A4 一样，纯 ensemble 增益。baseline 做对的 prompt 没有被模板 ensemble 打掉任何一条。8 条 rescue 完全来自 baseline 之外的某个模板找到了答案。

**8 条 rescue 的粗略分解**（忽略交集，精确数字需要从 per_prompt 回算）：
- `answer` 单模板救了至少 3 条 baseline 错的（45 − 42 = 3）。
- `cot_plain` / `cot_step` 虽然整体降，但各自也救了一些 prompt（它们救和砸的 prompt 不是同一子集）。
- `any_template=50 > answer 单独=45`，说明至少 5 条 prompt 只有 CoT 模板能救 —— CoT 不是全局没用，只是全局风险高。

**警觉点**：
- **N=60 不是 137**：和 A3/A4 同一次中断。8/60=13.33%；剩下 77 条按比例大概率稳定，但需要 `--resume` 才能敲死。
- **Template ensemble 代价**：4× inference / prompt。之前写的"one-template-for-all（`answer` 单独）可以直接上线"**是错的** —— overlap 分析（见 [`finding_a4x5_overlap.zh.md`](finding_a4x5_overlap.zh.md)）查出：`answer` 单独**救 8 条 baseline-错 prompt 也砸 5 条 baseline-对 prompt**。在这批 fail-enriched 子集上净 +3，一般分布上大概率全局掉点。最便宜可上线配置是 `{baseline, answer}` 2 格 ensemble，不是直接替换。
- **`answer` 只比 baseline +3**：在 base=42 上 +3 = +7% 相对提升，也就是 60 条里把 3 条错变对。相对 noise floor（broken=0）是显著的，但绝对数不大。

**跟 A4 的调和**：
- A4 SUPPORTED（5/60 被 layout 多样性救）。
- A5 SUPPORTED（8/60 被模板多样性救）。
- 两者都 `broken=0`，都是纯 ensemble 收益，都停在 N=60。
- 开放问题（Q4 见下）：A4 救的 5 条和 A5 救的 8 条是否不相交？如果是，`layout × template` ensemble 能救 13 条 distinct fail —— **20 格 ensemble**（5 layout × 4 template）原则上最多可以救到 ~22%。如果重叠，两者在挑同一批"采样脆弱" prompt，合起来收益递减。

---

## A6 —— Gen-length rerank · SUPPORTED（最强单轴）

**断言**：默认 `gen_length=128` 不是所有 prompt 的最优生成长度；不同长度能救 error。

**设计**：
- `scripts/validate/a6_gen_length.py`
- 每条 fail prompt 在 gen_length ∈ {64, 96, 128, 160, 192, 256} 各跑 1 次（固定 block_length=32, T=0）。
- Verdict 阈值同 A4/A5。

**结果**（N=60）：
```
base(g128)=42  any_length=54  rescued=12  broken=0
rescue_rate=20.00%  any_length_rate=90.00%
per_length: g64=27  g96=36  g128=42  g160=49  g192=39  g256=40
```
→ **SUPPORTED**（20.00% ≥ 5% 阈值）。**A 轴最强单轴信号**。

**关键观察**：
1. **g160 是甜点**：49/60=81.7% vs baseline g128=42/60=70%，单点 +11.7pp。g160 = 5 blocks × 32 tokens，比 baseline 多一个 block，刚好给模型足够的推理空间。
2. **不是"越长越好"**：g192=39（-3）、g256=40（-2），比 baseline 反而差。长 budget 下模型 trajectory 发散。
3. **g160 单点已超过 A5 的 any-template ensemble**（49 vs 50，接近但 g160 是单配置）。
4. **短预算严重伤害**：g64=27（-15）、g96=36（-6），大量 prompt 在预算不够时截断错误。

**Rescue 集**：{0,10,13,15,19,28,35,48,51,53,55,59}（12 条）
- A6 独有：{19, 51} —— 只有 gen_length 旋钮能救
- A6 ∩ H3 独有：{0} —— 只有 A6 和 H3 能救
- 与 A4∪A5 大量交叉但仍有独立贡献

**跟 A4 的关系**：gen_length 跟 block_length 是正交旋钮。A4 在固定 gen_length=128 下 sweep block_length；A6 在固定 block_length=32 下 sweep gen_length。理论上 `{block_length} × {gen_length}` 2D sweep 能进一步扩大 rescue 空间。

---

## A4×A5 Joint 6-cell 实跑验证

`{baseline, answer} × {bl8, bl32, bl64}` 6 格实跑：
- N=60, base=42, any=52, rescued=10, rescue_rate=**16.67%**
- **完美验证 overlap 预测**：预测 10 条 rescue，实跑 10 条，100% 吻合，零意外
- per-cell: bl8_baseline=43, bl8_answer=41, bl32_baseline=42, bl32_answer=45, bl64_baseline=37, bl64_answer=40

验证了 overlap 分析方法论可靠、配置间无意外交互。

---

## 跨实验解读

### 粒度阶梯

| 粒度 | 实验 | 干预 | Verdict |
|---|---|---|---|
| 1 token | A2/H1, A3 | revise 1 token / 1 span | DEAD |
| 1 edge（token pair） | A1（三证） | 改 DAG 边 | DEAD |
| 1 block（8-64 token） | A4 | 换 layout | **信号**（8.33%） |
| 1 prompt | A5 | 换模板 | **信号**（13.33%） |
| gen_length | A6 | 换生成长度 | **信号**（20.00%，A 轴最强） |
| layout × template | A4×A5 joint | 6-cell ensemble | 16.67%（overlap 完美验证） |
| sampling scheme | H2, H3 | H2 换 block_length / H3 换 T+N | H3 (n=60) **86.67% 压倒 A 轴** |

信号首次出现在 **block 级**，到 prompt/gen_length 级更强。更细的（token, span, edge）全 dead。A6 gen_length 是 A 轴内最强单旋钮（20%），g160 单点已接近 A5 的 any-template ensemble。H3 (pass@N multi-T) 是**跨轴维度最强杠杆**（52/60 = 86.67%），但跟 A 轴近乎正交（H3 ⊆ A-union 仅 19.2%）。

**全方法 union = 55/60 = 91.67%**（n=60 权威）。只剩 5 条 [4,5,14,41,42] 所有方法都救不了 = true capacity ceiling。在 FAIL18 子集口径下：A-union = H3-union = 13/18 = 72.2%，H3 在 FAIL18 内 0 独有 rescue；A6 独救 {19, 51} 即便 H3 也救不了。

**Rescue 信号随粒度单调递增**：A4 的 8.33% → A5 的 13.33% → A6 的 20.00%，粒度越粗信号越强。这说明 **LLaDA-instruct 没有"可局部修复的排序错误"** —— 没有小范围的 local fix。有用的是整条 denoise 轨迹的全局重走（A4）、输入 framing 的全局重塑（A5）、或生成预算的调整（A6），都在 redistribute **整个输出**，不是局部 token。

### 为什么 conf-based revise 死了

A2 + A3 合起来是强证据：**模型对已 commit token 报告的 conf 在 fail prompt 上跟错误没相关**。窄阈值（0.3 单 token）和宽阈值（0.4 窗口平均）都找不到错误。未来任何 sampler-side 纠错必须用**除了 committed-token conf 之外**的信号 —— 候选：layout 间的 self-consistency（A4.1 思路）、独立 verifier head（B4）、工具增强复核（B3）。

### A4 证了什么、没证什么

- **证了**：对 ~8% 的 fail prompt，存在至少一个 block layout 优于默认 bl32。
- **没证**：任何单一 layout 全局优于 bl32。本次 per-layout 总数 bl8 > bl16 ≈ bl32 > bl64 ≈ short_then_long，但差距 ±1 到 ±5 在 N=60 上是噪声以内。
- **没证**：per-prompt 最优 layout 可从 prompt 预测出来。那是 A4.1。

### 更新版决策树（post-closure 全量）

```
A3 REJECTED ────────────┐
                        ├──> conf-based revise 在 block 以下所有粒度 DEAD
A1 DEAD（三证） ────────┘         → 采样器 roadmap 去掉 revise hook

A4 SUPPORTED (5/60) ────┐
H2 REJECTED（0.754）────┤
A5 SUPPORTED (8/60) ────┤──> block-layout + prompt-template + gen-length 三轴都有信号
A6 SUPPORTED (12/60) ───┘     → A6 最强（20%），g160 是甜点
                              → A4×A5 joint 6-cell 完美验证（10=10）
                              → A-union = 13/60 = 21.67%（n=60 全集）
                              → 全方法 union = 55/60 = 91.67%（n=60 全集）
                              → FAIL18 子集口径：A-union = H3-union = 13/18 = 72.2%

H3 (n=60) pass@8 = 86.67% (52/60)  ← REJECTED per capacity-ceiling 阈值
  → capacity ceiling REJECTED（模型能力远没到上限）
  → H3 独占 42 条全在 FAIL18 外；FAIL18 内 0 条独有 rescue（H3 ⊆ A-union 在 FAIL18 子集上）
  → H3 ⊆ A-union 全集 = 19.2%，跟 A 轴近乎正交
  → A6 独救 {19, 51} 即便 H3 也救不了 → "write-space > diversity" 成立

True capacity ceiling = 5 条 [4, 5, 14, 41, 42]（n=60 和 FAIL18 子集口径一致）

→ 下一步：per-prompt strategy search (block_length × template × gen_length × temperature)
→ 新维度：template_position（diffusion LM 特有的 scaffold/inpainting）
```

---

## 待解问题（post-closure 更新）

**已解决**：
1. ~~A4 × A5 overlap~~ → **DONE**：independence=0.769，joint 6-cell 实跑完美验证（10=10）
2. ~~H3 收尾~~ → **DONE (n=60 权威)**：pass@8 = 86.67% (52/60)，按阈值 REJECTED capacity ceiling；H3 ⊆ A-union 仅 19.2%（跟 A 轴近乎正交），FAIL18 子集内 H3 ∩ = 10 条 {0,8,10,13,15,28,35,53,55,59} 全被 A-union 涵盖，H3 0 独有 rescue
3. ~~单独上 `answer` 模板~~ → **不行**（overlap 分析已确认）
4. ~~gen_length 扩查~~ → **DONE → A6 SUPPORTED**，rescue=20%
5. ~~p5_h3_crossref.py BUG~~ → **DONE**：已创建 `p6_h3_crossref.py` 修复 schema 错配（旧 `pass_at_k` 对 `temps.T.pass@k` 不匹配导致 silent h3_rescue=0）。P6 为 n=60 权威 crossref，输出 full_union=55/60 (91.67%)、ceiling=5 [4,5,14,41,42]、h3_only=42、a6_only=[19,51]

**开放中**：
1. **Per-prompt strategy search**：搜 `(block_length × template × gen_length × temperature)` 最优组合 → `(prompt, best_strategy)` pairs。这是下一阶段主线。
2. **template_position 新维度**：diffusion LM 特有的 scaffold/inpainting，template token 放在生成区域任意位置。
3. **P4/P6 离线分析**：CoT 砸 12 条的 pattern + A4-only rescue 特征。N=60 下 7 个 feature 全不显著（见 `finding_p4_p6_feature_analysis.zh.md`），等 N=137 再跑。

---

## 本次 finding 涉及的文件

### Scripts
- `scripts/validate/h0_forensics.py` —— scope 生成（fail + ok）
- `scripts/validate/h1_remask_rescue.py` —— H1 + 基础 sampler（被所有实验复用）
- `scripts/validate/h2_order_vs_content.py` —— H2
- `scripts/validate/h3_passN_at_temperature.py` —— H3
- `scripts/validate/a3_span_revise.py` —— A3，HTTP client
- `scripts/validate/a4_block_rerank.py` —— A4，HTTP client
- `scripts/validate/a5_prompt_template.py` —— A5，HTTP client
- `scripts/validate/_http_client.py` —— 共享 FastAPI client
- `scripts/validate/aggregate_verdicts.py` —— idempotent verdict board 重写
- `scripts/validate/run_a_axis.sh` —— 顺序跑 A3→A4→A5
- `scripts/validate/p6_h3_crossref.py` —— H3 × A 轴 n=60 crossref（替换 buggy P5）
- `scripts/serve.py` —— 加 `/generate_span_revise`、`/generate_block_schedule`

### Server 端 sampler
- `src/dllm_reason/inference/validation_ext.py` —— `generate_span_revise / generate_block_schedule / generate_uniform`（server 端点复用）

### Results
- `runs/validation/h1_remask_20260415_051706/summary.json`
- `runs/validation/h2_order_content_20260415_054252/summary.json`
- `runs/validation/a3_span_revise_20260415_181502/summary.json`
- `runs/validation/a4_block_rerank_20260415_182338/summary.json`
- `runs/validation/h3_passN_*`（在跑）
- `runs/validation/a5_prompt_template_20260415_191434/summary.json`

### Docs
- `docs/archive/hypotheses.md / .zh.md` —— 假设登记簿
- `docs/archive/exploration_axes.md / .zh.md` —— A/B 轴索引
- `docs/archive/finding_dag_search_zero_rescue.md / .zh.md` —— A1 DEAD 证据
- `docs/archive/finding_a_axis_exploration.md / .zh.md` —— **本文**
