# [RETRACTED] `answer` 砸的 5 条是数字巧合 —— 证据是伪造的

> 语言：中文
> 原日期：2026-04-15 · 撤回日期：2026-04-15
> 原文件名：`finding_broken_by_answer_is_spurious.zh.md`（已 rename 为本文件）
> 撤回证据：`runs/validation/p21e_full_broken5_g256.json`

---

## TL;DR —— 这份 finding 完全错

原 finding 声称：A5 run 里 5 条 broken-by-answer 的 baseline `correct=True` 是"虚假的数字巧合"，模型输出的是**另一个完全无关** gsm8k 题的解答，只是最后数字撞上 gt。

**真相**：原 finding 引用的 "prompt" 和 "baseline output 主题" 全是我手误/乱引的。A5 这 5 条 baseline tail 本来就在**正确解答 prompt 自己的问题**，没有任何巧合也没有任何污染。

---

## 对照表

| idx | gt | 真实 prompt（从 `scope_fail_prompts.json[idx]` 核实） | 原 finding 错引的 prompt | baseline tail 实际在解什么 |
|---|---|---|---|---|
| 2 | 64 | **Kylar 买 16 glasses**，每第二个 60% 价 | ❌ "John 开车 3 小时..." | 正解 Kylar（$5+$3×8=$64） |
| 17 | 104 | **Gloria 买 boots/heels**，heels 33+66=99，比 boots 少 5 | ❌ "Jerry 泳池漏水..." | 正解 Gloria（99+5=104） |
| 22 | 48 | **薯片 300g/5 serv/250 cal**，剩 200 cal | ❌ "Rick 打猎..." | 正解薯片（0.8 serv × 60g = 48g） |
| 24 | 163 | **Candice Post-it**，80 + package − 220 = 23 | ❌ "煮油加热..." | 正解 Post-it（163） |
| 57 | 26 | **John 平日 4 杯水、周末 3 杯** | ❌ "Burger Palace 果冻豆..." | 正解 John（4×5+3×2=26） |

原 finding 里列的 John-driving / Jerry-pools / Rick-hunting / oil-heating / jelly-beans 这 5 条 prompt 都是**我混淆了的别条 gsm8k 题**，跟 scope_fail 的 idx=2/17/22/24/57 **毫无关系**。

验证方法：P2.1.e (`scripts/validate/p21e_dump_full.py`) 对这 5 条 idx 重跑 A5 baseline 并保存完整输出（不是 tail[-200:]）。输出里的 prompt 和 head 都明确显示这 5 条都是在正常解 prompt 自己的问题。

---

## 原 finding 的连锁影响（全部撤回）

| 原主张 | 状态 |
|---|---|
| `answer` 救 8 砸 5 = 3 条虚假正确 | ✅ 仍然对。`answer` 砸的 5 条是 baseline **真的对了**、answer 模板真的算错了 —— 是真 trade，不是数字巧合 |
| base_correct=42 里含"虚假正确" | ❌ 撤回。基础信号无污染 |
| A4/A5 rescue rate 可能虚高 | ❌ 撤回 |
| scope_fail_prompts 可能被误 classify | ❌ 撤回 |
| P2.1.a/b/c/d 内容 coherence judge | ❌ 全部取消，不用做 |
| 训练集泄露 / decoder fallback 假说 | ❌ 从未成立，原证据是幽灵 |

**相关有效结论不受影响**：
- A4 SUPPORTED `rescue_rate=8.33%` —— 有效
- A5 SUPPORTED `rescue_rate=13.33%` —— 有效
- A4 × A5 `independence_factor=0.769` —— 有效
- `{baseline, answer}` 2-cell ensemble net +3 —— 有效（不过下游意义仍取决于 ensemble 设计）

---

## 方法论教训 —— 保留

1. **诊断时一定要核对 prompt 文本本身**。per-prompt JSON 只存了 `tail`（`out[-200:]`），看 tail 想象 prompt 容易串线。
2. **"从记忆里列 5 条证据" 不能替代 read-file 核实**。本 finding 的崩塌就是我凭印象写了 5 条 prompt 内容，没回去验证 `scope_fail_prompts.json[idx].prompt`。
3. **A5 存 tail[-200:] 是真实的信息损失**。以后 rerun 应该保留完整输出（至少对诊断用的 fail set）—— 已在 P2.1.e 脚本里做到。

---

## 衍生观察（仍待跟进，独立于本撤回）

P2.1.e 的 g256 rerun 意外发现：**把 gen_length 从 128 提到 256，同一 baseline prompt、T=0 下 5 条里 4 条从正确变错**。这不在本 finding 范围，另起 `finding_gen_length_sensitivity.zh.md` 记录。

---

## 归档状态

**本 finding 整体撤回**。保留文件作为方法论反面案例。
