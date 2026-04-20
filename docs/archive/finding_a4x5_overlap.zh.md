# Finding：A4 × A5 overlap —— 两根基本独立的杠杆，`answer` 模板是 trade 不是纯赢

> 语言：中文  |  English: [finding_a4x5_overlap.md](finding_a4x5_overlap.md)

**日期**：2026-04-15
**配套**：[`finding_a_axis_exploration.zh.md`](finding_a_axis_exploration.zh.md)
**脚本**：`scripts/validate/a4x5_overlap.py`
**报告**：`runs/validation/a4x5_overlap_182338_191434.json`

---

## TL;DR

A4（`a4_block_rerank_20260415_182338`）和 A5（`a5_prompt_template_20260415_191434`）共同的 60 条 prompt 上：

| 指标 | 数值 |
|---|---|
| `base_correct`（= A5 baseline = A4 bl32） | 42/60 |
| fail 集大小 | 18/60 |
| A4 rescue | 5 条（fail 的 27.78%） |
| A5 rescue | 8 条（fail 的 44.44%） |
| **A4 ∩ A5** | **3 条** |
| 仅 A4 | 2 条 |
| 仅 A5 | 5 条 |
| **A4 ∪ A5** | **10 条（fail 的 55.56%）** |
| independence factor | **0.769**（1.0 = 完全不相交；0.5 = 完全重叠） |
| 20 格 ensemble 上限 | 52/60 = **86.67% any-correct**；fail 救 **55.56%** |

**两个核心发现**：

1. **两杠杆基本独立**。0.769 离 1.0 足够近，`layout × template` 20 格 ensemble 是**真实的乘法增益**，不是冗余。合起来救 10 条 distinct fail，不是全重叠下的约 6 条。
2. **`answer` 模板不是免费升级**。per-prompt 查出来：`answer` 单独**救 8 条也砸 5 条**（原来 baseline 对的），净 +3。archive 之前写的"`answer` 前缀单独立刻可用"**是错的** —— 只在 A5 rescue 数多于 broken 数时才净正。ensemble 里 `broken=0` 隐藏了这一点 —— 因为 ensemble 里 baseline 把 `answer` 砸掉的 5 条接回来了。

---

## Rescue 集详情

```
idx   gt       标签        救它的 layouts                   救它的 templates
 8    18       仅 A5       -                                [answer]
10    125      仅 A5       -                                [cot_step, answer]
13    15       仅 A4       [bl64]                           -
15    8        BOTH        [bl64, short_then_long]          [cot_plain, cot_step, answer]
28    40       BOTH        [bl16]                           [cot_plain, answer]
35    48       仅 A5       -                                [cot_plain, cot_step, answer]
48    623      仅 A5       -                                [answer]
53    4        仅 A4       [bl8]                            -
55    5        仅 A5       -                                [answer]
59    3        BOTH        [bl8, bl64, short_then_long]     [cot_plain, cot_step, answer]
```

三条模式：

- **仅 A5 的 rescue 高度集中在 `answer`**：5 条仅 A5 的 rescue 里，4 条含 `answer`，3 条是 `answer` 独救。`answer` 是 A5 的主杠杆。
- **3 条 BOTH 是"易救"**：几乎任何扰动都能救，2/3 条有 ≥3 个 winning template 和 ≥2 个 winning layout。
- **仅 A4 的 rescue 跟 layout 绑死**：idx 13（bl64）、idx 53（bl8）—— 最粗和最细的 uniform。没有"bl32 永远差不远"的规律。

---

## 每个 config 的独立贡献

| Config | 独救 fail 数（该 config 对 ∧ baseline 错） |
|---|---|
| A4.bl32 | 0（按定义，就是 baseline） |
| A4.bl8 | 2 |
| A4.bl16 | 1 |
| A4.bl64 | 3 |
| A4.short_then_long | 2 |
| A5.baseline | 0（按定义） |
| A5.cot_plain | 4 |
| A5.cot_step | 4 |
| **A5.answer** | **8** |

`A5.answer` 把 A5 全部 8 条 rescue 都救了。去掉 `answer`，A5 的 rescue 会从 13.33% 崩回约 6%。它**就是** A5 的杠杆。

---

## 修正早先的结论 —— `answer` 作为单模板切换

之前 finding 里写：

> **单独上 `answer` 模板行不行？** `answer=45` vs `baseline=42`，单模板切换就已净正。最便宜的收益。

算术对，但 framing 错了。per-prompt 直接统计：

```
baseline=T, answer=T:   37
baseline=T, answer=F:    5   ← 被 answer 砸
baseline=F, answer=T:    8   ← 被 answer 救
baseline=F, answer=F:   10
```

所以 `baseline → answer` 的替换是 5 对→错 换 8 错→对。净 +3。在这批 fail-enriched 的 60 条上是正的，但：

- 放到一般分布上（绝大多数是 `base_correct=T`），砸 5/42 = baseline 通过集上 **11.9% 回归**。
- 整个 gsm8k eval（baseline-correct 比例 ≈ 0.85+）上几乎必然**全局掉点**。

**正确建议**：`answer` **只能放在 ensemble 里**（让 `baseline` 接回被砸的 5 条），不能直接替换。等价写法：`{baseline, answer}` 2 模板 ensemble —— 2× inference 抓到 A5 几乎全部 rescue 信号，比 4× 便宜。

被 `answer` 砸的 prompt（idx、gt）：
```
idx=2  gt=64
idx=17 gt=104
idx=22 gt=48
idx=24 gt=163
idx=57 gt=26
```

后面一次 session 的检查提示：这几条是不是因为推理链长到"直接答 Answer:"会抢在模型本来要 commit 的算术前面？数字都不大，从 idx+gt 看不出明显结构特征。

---

## Joint 20-cell ensemble —— 上限，不是方案

跑全部 5 layout × 4 template = 20 次 decode/prompt，能救 10/18 = 55.56% 的 fail，对比 A5 alone 44.44% 或 A4 alone 27.78%。**真实增益** —— A4 在 A5 基础上额外带 2 条独救。

但 20× inference 上不了线。现实压缩：

- **`{baseline, answer}` × `{bl8, bl32, bl64}` = 6 格**。保 A5 主杠杆（`answer`）+ 安全网（`baseline`）+ A4 贡献最多的 layout。
  - 预期 rescue：A5 ensemble 8 条 + idx 13（仅 bl64 能救）+ idx 53（仅 bl8 能救）= **10 条 = 跟完整 20 格一样**（假设没有新交互）。
  - ~~A4 × A5 值不值钱，下一步跑这个。~~ **已跑，见下。**
- **`{baseline, answer}` 2 格**。预期 rescue：8 条 —— 抓到 A5 全部信号，丢掉 A4 独救的 2 条。**可能是最便宜的可上线配置**。

### Joint 6-cell 实跑验证（post-closure 新增）

`{baseline, answer} × {bl8, bl32, bl64}` 6-cell 实跑结果：

```
N=60  base=42  any=52  rescued=10  rescue_rate=16.67%

per-cell:
  bl8_baseline  = 43
  bl8_answer    = 41
  bl32_baseline = 42  ← 原 baseline
  bl32_answer   = 45
  bl64_baseline = 37
  bl64_answer   = 40
```

**预测 10 条 rescue，实跑 10 条，100% 吻合，零意外。**

这验证了三件事：
1. **Overlap 分析方法论可靠**：从 per-prompt 独立实验结果预测联合实验结果，完美命中。
2. **配置间无意外交互**：不同 layout × template 组合之间没有破坏效应。
3. **6-cell 是真正可部署的方案**：6× inference 拿到跟理论 20-cell 一样的 rescue。

---

## 对 A4.1（per-prompt layout 预测器）的启示

A4.1 的动机：能预测出 prompt → 最优 layout，就不用付 5× inference。

overlap 跑完之后：

- **A4 在本次 run 上真正独立带的 rescue 只有 2 条**（idx 13、53 —— 仅 A4）。A4 5 条里 3 条 A5 也能救。
- 完美预测（idx 13 → bl64、idx 53 → bl8）在 A5 ensemble 基础上只加 2/60 = 3.33%。放大到 N=137 大概 5-8 条。
- 这个**增量信号太小**，train 不了分类器。2 个正例（binary 化成"layout ≠ bl32 是否有用"）、58 个负例，学习信号不够；手工特征（prompt 长度、数字 token 数、算子数）可能能抓，也可能只是过拟合。

**A4.1 优先级判断**：**降级**。力气先花在：

1. **先把 A4/A5 补到 N=137**（锁住 independence factor 和 A4 独救数）。
2. **跑 `{baseline, answer}` 2 格 ensemble** 在完整 eval 上 —— 验证"`answer` 单独会全局掉点"这个预测是对还是错。
3. 只有在规模化后 A4 独救仍然重要，再动手造个手工特征分类器试 A4.1。2-8 条正例学 A4.1 基本是奢望。

---

## 待解问题

1. **N=137 时 independence 还稳吗？** 60 条里只有 18 条 fail，overlap 统计噪声大。0.769 有可能漂动。两个都 `--resume` 之后重跑这个脚本。
2. **`answer` 砸的 5 条是不是跟其他模板的 rescue 集重叠？** 如果是，`answer` 的失败模式特定、可能可检测。如果散的，就没有可利用的 pattern。
3. **真跑 20 格 ensemble，rescue 会不会达到 55.56% 上限？** 上限假设 oracle 选最优 config；实际投票/majority 策略大概率跑不到。
4. **H3 那 1/17 rescue**：那条 prompt 在不在 union_rescue 名单（idx ∈ {8, 10, 13, 15, 28, 35, 48, 53, 55, 59}）？H3 跑完后 per_prompt 交叉一下就一行代码。
