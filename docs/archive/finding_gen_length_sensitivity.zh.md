# Finding: A6 gen_length rerank — A 轴最强单轴信号

> 语言：中文
> 日期：2026-04-15（初始 N=5 观察）→ 更新至 post-closure 全量结果
> 状态：**SUPPORTED** —— N=60, rescue_rate=20.00%, 最强单轴信号
> 来源：P2.1.e rerun（初始观察）→ `scripts/validate/a6_gen_length.py`（扩查）

---

## 观察

P2.1.e 原本是为撤回 `retracted_broken_by_answer.zh.md` 而 rerun，目的是 dump 完整输出。顺手对比了 **g128 (A5 原 run) vs g256 (本次 rerun)**，同一 prompt、同一 baseline 模板、T=0、block_length=32：

| idx | prompt 主题 | gt | g128 baseline | g256 baseline |
|---|---|---|---|---|
| 2 | Kylar 眼镜 | 64 | ✅ 64 | ❌ 24 |
| 17 | Gloria heels/boots | 104 | ✅ 104 | ✅ 104 |
| 22 | 薯片卡路里 | 48 | ✅ 48 | ❌ 0 |
| 24 | Candice Post-it | 163 | ✅ 163 | ❌ 23 |
| 57 | John 喝水 | 26 | ✅ 26 | ❌ 24 |

**5 条里 4 条在 g256 下从正确翻转成错误**。T=0 下 deterministic，不是 noise。

---

## 为什么这是信号

- g128 = 4 blocks × 32 tokens；g256 = 8 blocks × 32 tokens
- A4 之前只 sweep 了 `block_length ∈ {8, 16, 32, 64}`（固定 gen_length=128），**total length 作为 layout 的一部分没被扫**
- A4 rescue_rate=8.33% 可能低估了 A 轴能挖的空间 —— 换 gen_length 等价于换 block 数量，理论上是 A4 的正交旋钮

## 解释候选

- **(1) 越长越乱**：diffusion LM 在更多 block 下需要协调的依赖更多，T=0 下更易跑偏到错误 trajectory
- **(2) LLaDA SFT 偏短答**：Instruct 训练里多数样本 answer 较短，长 budget 下分布不稳
- **(3) block 边界扰动**：8 个 block 的 commit 顺序跟 4 个 block 不同，导致关键 reasoning token 被 commit 到不同位置

## 反方向也要查

**逆检验**：g64 下这 5 条表现如何？
- 如果 g64 正确率 > g128 → 越短越好，LLaDA 偏爱短答案 (2)
- 如果 g64 ≈ g128 → g=128 是个局部最优，(1) 或 (3) 更可能
- 如果 g64 < g128 → g=128 正好在 sweet spot，意味着 gen_length 有个最优值，是 A 轴的另一把可调旋钮

---

## 扩查结果（N=60） —— SUPPORTED

扩查已完成。结果远超预期：

```
N=60  base(g128)=42  any_length=54  rescued=12  broken=0
rescue_rate=20.00%  any_length_rate=90.00%

per-length:
  g64  = 27/60 (45.0%)
  g96  = 36/60 (60.0%)
  g128 = 42/60 (70.0%)  ← baseline
  g160 = 49/60 (81.7%)  ← 甜点
  g192 = 39/60 (65.0%)
  g256 = 40/60 (66.7%)
```

→ **SUPPORTED**（20.00% ≥ 5% 阈值）。**A 轴最强单轴信号**。

### 关键发现

1. **g160 是甜点**：49/60=81.7%，比 baseline g128 +11.7pp。g160 = 5 blocks × 32 tokens，多一个 block 刚好给模型足够的推理空间。
2. **不是"越长越好"**：g192/g256 回落到 baseline 以下。初始 N=5 的"g256 翻转"观察在 N=60 上被确认为整体趋势。
3. **rescue_rate=20%** 是 A4（8.33%）的 2.4 倍，A5（13.33%）的 1.5 倍。
4. **Rescue 集**：{0,10,13,15,19,28,35,48,51,53,55,59}（12 条），与 A4/A5 有大量交叉但仍有 2 条独有（{19,51}）。

### 初始 N=5 观察的验证

初始观察"g256 下 4/5 翻转"在 N=60 上得到了更全面的图景：g256 整体确实比 g128 差（40 vs 42），但差距不大（-3.3pp）。真正的信号不在"g256 坏"，而在"g160 好"。

---

## 下一步（已被 strategy search 吸收）

原计划的 A4 × A6 2D sweep 被整合进 per-prompt strategy search pipeline（搜 `block_length × template × gen_length × temperature` 全组合）。gen_length 作为最强单轴，是 strategy search 的核心维度。

---

## 历史：初始 N=5 范围警告

~~N=5 的观察不能当结论。~~ 已被 N=60 扩查确认并升级为 SUPPORTED。
