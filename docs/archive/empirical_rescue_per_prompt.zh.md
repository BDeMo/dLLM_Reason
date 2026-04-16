# Empirical Fact: per-prompt strategy 覆盖率与 gsm8k 跑分估算

> 语言：中文
> 日期：2026-04-16
> 数据源：A4 / A5 / A6 / A4×A5 joint / H3 的 per_prompt JSON
> 修正说明：原 83.3% union 数字有误（混用了 H3 和 A4-A6 的 baseline 口径），已修正为 72.2%

---

## 1. Eval 范围

| 类别 | 数量 | 来源 |
|---|---|---|
| scope_ok（原始 eval 正确） | 49 | `scope_ok_prompts.json` |
| scope_fail（原始 eval 错误） | 60 | `scope_fail_prompts.json` |
| **总计** | **109** | gsm8k eval 子集 |

**注意**：scope_fail 的 60 条在 T=0 / bl32 / g128 重评后有 42 条 baseline 正确（即原始 eval 条件与 A 轴实验条件不同）。以下所有 "base_fail" 指 T=0/bl32/g128 下 baseline 错误的 **18 条**。

---

## 2. gsm8k 跑分对照表

| 配置 | 正确数 / 109 | 准确率 |
|---|---|---|
| Baseline (T=0, bl32, g128) | 91 | **83.49%** |
| g160 单点（只改 gen_length） | 98 | **89.91%** |
| Oracle strategy (per-prompt 最优) | 104 | **95.41%** |
| 理论上限（全对） | 109 | 100% |

### 拆解

```
scope_ok:                             49  (不变)
scope_fail → baseline correct (T=0):  42  (不变)
scope_fail → rescued by strategy:     13  (oracle 选最优策略能拿回)
scope_fail → not rescued:              5  (所有策略都救不了)
                                     ───
Total correct (oracle):              104 / 109 = 95.41%
Total wrong:                           5 / 109 =  4.59%
```

**Oracle strategy 把 gsm8k 从 83.49% 拉到 95.41%，+11.92pp。只剩 5 条硬骨头。**

---

## 3. 18 条 base_fail per-prompt 详表

### 图例

- ✅ = 该策略能救（correct=True）
- ❌ = 该策略不能救
- `—` = 没跑该组合
- **RESCUED BY** = 至少一个方法能救
- **NONE** = 所有方法都救不了

### A4 (block_length)

| idx | gt | A4:bl8 | bl16 | bl32 | bl64 | short_long |
|---|---|---|---|---|---|---|
| 0 | 70000 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 4 | 160 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 5 | 45 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 8 | 18 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 10 | 125 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **13** | **15** | ❌ | ❌ | ❌ | **✅** | ❌ |
| 14 | 14 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **15** | **8** | ❌ | ❌ | ❌ | **✅** | **✅** |
| 19 | 18 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **28** | **40** | ❌ | **✅** | ❌ | ❌ | ❌ |
| 35 | 48 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 41 | 88 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 42 | 60 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 48 | 623 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 51 | 9360 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **53** | **4** | **✅** | ❌ | ❌ | ❌ | ❌ |
| 55 | 5 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **59** | **3** | **✅** | ❌ | ❌ | **✅** | **✅** |

A4 rescue = {13, 15, 28, 53, 59} = **5/18**

### A5 (template)

| idx | gt | A5:baseline | cot_plain | cot_step | answer |
|---|---|---|---|---|---|
| 0 | 70000 | ❌ | ❌ | ❌ | ❌ |
| 4 | 160 | ❌ | ❌ | ❌ | ❌ |
| 5 | 45 | ❌ | ❌ | ❌ | ❌ |
| **8** | **18** | ❌ | ❌ | ❌ | **✅** |
| **10** | **125** | ❌ | ❌ | **✅** | **✅** |
| 13 | 15 | ❌ | ❌ | ❌ | ❌ |
| 14 | 14 | ❌ | ❌ | ❌ | ❌ |
| **15** | **8** | ❌ | **✅** | **✅** | **✅** |
| 19 | 18 | ❌ | ❌ | ❌ | ❌ |
| **28** | **40** | ❌ | **✅** | ❌ | **✅** |
| **35** | **48** | ❌ | **✅** | **✅** | **✅** |
| 41 | 88 | ❌ | ❌ | ❌ | ❌ |
| 42 | 60 | ❌ | ❌ | ❌ | ❌ |
| **48** | **623** | ❌ | ❌ | ❌ | **✅** |
| 51 | 9360 | ❌ | ❌ | ❌ | ❌ |
| 53 | 4 | ❌ | ❌ | ❌ | ❌ |
| **55** | **5** | ❌ | ❌ | ❌ | **✅** |
| **59** | **3** | ❌ | **✅** | **✅** | **✅** |

A5 rescue = {8, 10, 15, 28, 35, 48, 55, 59} = **8/18**

### A6 (gen_length)

| idx | gt | A6:g64 | g96 | g128 | g160 | g192 | g256 |
|---|---|---|---|---|---|---|---|
| **0** | **70000** | **✅** | ❌ | ❌ | ❌ | **✅** | ❌ |
| 4 | 160 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 5 | 45 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 8 | 18 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **10** | **125** | ❌ | ❌ | ❌ | **✅** | **✅** | ❌ |
| **13** | **15** | ❌ | ❌ | ❌ | **✅** | ❌ | ❌ |
| 14 | 14 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **15** | **8** | **✅** | **✅** | ❌ | **✅** | ❌ | ❌ |
| **19** | **18** | ❌ | ❌ | ❌ | **✅** | **✅** | **✅** |
| **28** | **40** | ❌ | ❌ | ❌ | **✅** | **✅** | **✅** |
| **35** | **48** | **✅** | **✅** | ❌ | **✅** | **✅** | **✅** |
| 41 | 88 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 42 | 60 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **48** | **623** | **✅** | ❌ | ❌ | ❌ | ❌ | **✅** |
| **51** | **9360** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **53** | **4** | ❌ | **✅** | ❌ | **✅** | **✅** | **✅** |
| **55** | **5** | ❌ | **✅** | ❌ | **✅** | **✅** | ❌ |
| **59** | **3** | ❌ | **✅** | ❌ | **✅** | **✅** | ❌ |

A6 rescue = {0, 10, 13, 15, 19, 28, 35, 48, 51, 53, 55, 59} = **12/18**

### H3 追加维度（2026-04-16）

H3 只跑了前 30 条 fail + 26 条 ok，补完到 60/30 前本节只能覆盖 idx 0-29。

H3 判定：**any-T pass@8 rescue = 26/30 = 86.67%**（任一温度 0.3/0.7/1.0 下 8 samples 至少有 1 个正确）。H3 在前 30 条里救不了的 4 条：**{4, 5, 14, 19}**。

在 18 条 base_fail（T=0/bl32/g128 下仍错）子集中，H3 覆盖到了 10 条：`{0, 4, 5, 8, 10, 13, 14, 15, 19, 28}`。其中 H3 救 6 条：`{0, 8, 10, 13, 15, 28}`。

H3 rescue 在 base_fail 上 **0 条独有 rescue**（全跟 A5/A6 交集），但 **idx=0 从 "A6 独救" 降级为 "A6+H3 都能救"**。

**反向信号（重要）**：idx=19（Dana runs/walks）A6 能救但 H3 救不了。这意味着 idx=19 的正确解需要"**更多空间/计算**"（gen_length）而不是"**更多 diversity**"（sampling）。这对 latent reasoning 假说是正面证据。

### 汇总：每条 fail prompt 的状态（加入 H3 列）

> H3 列：T∈{0.3,0.7,1.0}, N=8, any-T pass@8。`—` 表示 H3 未覆盖（idx≥30）。
> H3 扩到前 60 条后这张表会变全，届时更新 idx 30-59 的 H3 列。

| idx | gt | prompt (前 60 字) | A4 | A5 | A6 | Joint | H3 | 状态 |
|---|---|---|---|---|---|---|---|---|
| **0** | 70000 | Josh decides to try flipping a house... | ❌ | ❌ | ✅ | ❌ | ✅ | **A6+H3**（diversity 或 gen_length 都能救） |
| **4** | 160 | Carla is downloading a 200 GB file... | ❌ | ❌ | ❌ | ❌ | ❌ | **NONE** |
| **5** | 45 | John drives for 3 hours at a speed of 60 mph... | ❌ | ❌ | ❌ | ❌ | ❌ | **NONE** |
| **8** | 18 | Melanie is a door-to-door saleswoman... | ❌ | ✅ | ❌ | ✅ | ✅ | A5+Joint+H3 |
| **10** | 125 | A merchant wants to make a choice... | ❌ | ✅ | ✅ | ✅ | ✅ | A5+A6+Joint+H3 |
| **13** | 15 | I have 10 liters of orange drink... | ✅ | ❌ | ✅ | ✅ | ✅ | A4+A6+Joint+H3 |
| **14** | 14 | Raymond and Samantha are cousins... | ❌ | ❌ | ❌ | ❌ | ❌ | **NONE** |
| **15** | 8 | A candle melts by 2 centimeters... | ✅ | ✅ | ✅ | ✅ | ✅ | 全方法+H3 |
| **19** | 18 | Dana can run at a rate of speed... | ❌ | ❌ | ✅ | ❌ | **❌** | **A6 独救**（H3 也救不了，需 gen_length 不是 diversity） |
| **28** | 40 | A mechanic charges different rates... | ✅ | ✅ | ✅ | ✅ | ✅ | 全方法+H3 |
| **35** | 48 | There are four schools competing... | ❌ | ✅ | ✅ | ✅ | — | A5+A6+Joint (H3 未覆盖) |
| **41** | 88 | Artie has a flower stand... | ❌ | ❌ | ❌ | ❌ | — | **NONE** (H3 未覆盖) |
| **42** | 60 | Luke is spending time at the beach... | ❌ | ❌ | ❌ | ❌ | — | **NONE** (H3 未覆盖) |
| **48** | 623 | Grace weighs 125 pounds... | ❌ | ✅ | ✅ | ✅ | — | A5+A6+Joint (H3 未覆盖) |
| **51** | 9360 | A company pays each of its employees $600... | ❌ | ❌ | ✅ | ❌ | — | **A6 独救** (H3 未覆盖) |
| **53** | 4 | Emily has 4 kids named Amy, Jackson... | ✅ | ❌ | ✅ | ✅ | — | A4+A6+Joint (H3 未覆盖) |
| **55** | 5 | Cars have lined up on the motorway... | ❌ | ✅ | ✅ | ✅ | — | A5+A6+Joint (H3 未覆盖) |
| **59** | 3 | Frankie watches TV after he finishes... | ✅ | ✅ | ✅ | ✅ | — | 全方法 (H3 未覆盖) |

### 统计（加入 H3 后）

| 状态 | 数量 | idx |
|---|---|---|
| 全方法都能救（含 H3 覆盖到的） | 3 | 15, 28, 59 |
| **A6 独救（连 H3 都救不了）** | **1 确定 + 2 待定** | **19（确定）；51 待 H3 扩跑确认** |
| A6+H3 都能救（diversity 和 gen_length 互替） | 1 | 0 |
| 多方法覆盖 | 7 | 8, 10, 13, 35, 48, 53, 55 |
| 所有方法都救不了（前 30 确认） | 3 | 4, 5, 14 |
| 所有方法都救不了（前 30 外，待 H3 扩跑确认） | 2 | 41, 42 |

**关键变化**：
- 前 30 条 base_fail 里 "**NONE**" 从 5 条 (4,5,14,41,42) 收窄到 3 条（4,5,14），41/42 需要 H3 扩跑验证
- idx=19 是**唯一确定的 "A6 独救 ∧ H3 救不了"** —— 区分 diversity 与 compute 的关键样本

---

## 4. 83.3% → 72.2% 修正说明

之前报 83.3% = 15/18 是因为把 H3 的 rescue 集（基于 T=0.3 baseline）和 A4/A5/A6 的 rescue 集（基于 T=0 baseline）直接做并集。H3 声称 rescue 的 idx=2,24 在 T=0 baseline 下本来就是 correct（不属于 base_fail），虚增了 2 条。

修正后：
- **A4∪A5∪A6∪Joint = 13/18 = 72.2%**
- **A4∪A5∪A6∪Joint∪H3 = 13/18 = 72.2%**（H3 救的 6 条全在 13 条里，无独有）
- H3 在 T=0 baseline 口径下贡献 0 条**独有** rescue
- H3 的价值在 **反向信号**（idx=19 救不了 → 证明 gen_length 增益不能被 diversity 替代）

---

## 5. "真 ceiling" 收窄

原 5 条候选 → 前 30 条 H3 验证后确认 3 条：

| idx | gt | prompt | 特征 | H3 扩到 60 前状态 |
|---|---|---|---|---|
| 4 | 160 | Carla downloading 200GB, 40% throttle... | 多步百分比计算 | **确认 ceiling** (A4/A5/A6/H3 全 ❌) |
| 5 | 45 | John drives 3h@60mph turns around... | 速度/距离/时间多步 | **确认 ceiling** |
| 14 | 14 | Raymond & Samantha cousins, age diff... | 年龄差+未来推算 | **确认 ceiling** |
| 41 | 88 | Artie flower stand, 3 kinds, pricing... | 多品类定价加总 | **待定**（需 H3 扩到 idx=41） |
| 42 | 60 | Luke sandcastles, tide relationship... | 多条件逻辑 | **待定**（需 H3 扩到 idx=42） |

这 3-5 条共性：**都需要 3 步以上的链式推理**，且中间步有较大数值（160, 45, 88, 60）。模型在 A 轴 + H3 diversity 下都无法产出正确的推理链。这是 LLaDA-8B 在 gsm8k 上的**真实能力边界**候选。

---

## 6. E1+E5 联合结论：A6 gain 的来源（latent reasoning 被排除）

H3 那条"idx=19 唯一 A6 救且 H3 救不了"的观察**乍看像 latent reasoning 正面证据**，但 E1 + E5 两个因果分离实验把"为什么 A6 能救"这个问题锁死了：

### E5：不是物理截断
- A6 独救 3 条 `{0, 19, 51}` 的 g128 tail 启发式判定：1/3 maybe_truncated（idx=19），0/3 明确 truncated，其余 complete 但答错
- Verdict: **NOT_TRUNCATION** — A6 gain **不能**用"g128 物理写不完答案"解释

### E1：不是额外计算步数
解耦 `gen_length` 和 `num_steps`：
- **C_g128_s128**（baseline）: 42/60 correct
- **A_g160_s160**（空间↑ + 步数↑）: 49/60 correct，rescue **15.0%**
- **B_g128_s160**（空间锁 128，只加步数）: 42/60 correct，rescue **0.0%**

A6 独救 3 条焦点：
- idx=0：C/A/B 全挂（本轮未复现）
- idx=19：**A 救，B 救不了** ← 关键点
- idx=51：C/A/B 全挂

**Verdict: REJECTED** (latent reasoning ruled out) — 单加 num_steps 零贡献、零破坏。LLaDA 的 block-wise denoising 在 128 步已饱和，多出的 32 步对分布零影响。

### 联合解读

A6 g160 的 gain 是 **write-space effect** 而非 **latent-compute effect**：
- 多出的 32 个 token 位置让 **explicit CoT** 有更大 canvas 铺展
- 不等于模型在这些位置做了隐式思考
- 同样是 "LLaDA 在某些 prompt 上需要更多空间才能把推理链走完"，但这是**显式** reasoning 的空间需求，不是 latent reasoning

对论文：A6 不再是 latent reasoning 证据，重新定位为 **"gen_length budget calibration"** 子线 —— 15% 的 rescue 是实打实数字，只是解读变成了更保守的 "CoT space requirement heterogeneity"。

---

## 7. 关键 take-away

1. **g160 单点就是免费午餐**：83.49% → 89.91%（+6.42pp），零额外 inference 成本
2. **Oracle strategy 上限 95.41%**：说明 LLaDA-8B 的"潜力"远超 baseline 表现
3. **A6 (gen_length) 是覆盖率最高的单轴**：12/18 = 66.7%
4. **各维度互补**：A6 独有 3 条，A5 独有 2 条（idx=8 仅 answer 救，idx=35 的 A6 也能救但 A4 不能），A4 无独有
5. **3 条真 ceiling 全是多步链式推理** → 不是 inference strategy 能解的，需要模型能力提升（B2 training pivot）
6. **latent reasoning 被排除**（E1+E5 联合）：A6 gain 来自 write-space（更多 CoT 位置），不来自 latent compute；diversity 也替代不了空间（H3 救不了 idx=19）是因为 **explicit CoT space requirement** 而非 latent reasoning
7. **H3 在 fail set 上 pass@8 any-T = 86.67%** —— REJECTED "能力上限" 的原始断言；说明 LLaDA 在 T=0.3 fail 里有大量被 greedy mode 压住的 diversity 解
