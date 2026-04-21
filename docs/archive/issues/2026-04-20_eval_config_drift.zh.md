# Issue: v1.6 eval config drift from scope creation

**日期**：2026-04-20
**影响版本**：v1.6.0（定位 + 修 in v1.6.1）
**严重度**：🔴 **Major** —— 所有 v1.6.0 eval 数字**不可信**

---

## 症状

v1.6.0 首次 eval run 输出：

```
| Label     | fail pass@1    | ok pass@1       | FAIL18 rescued | ceiling broken |
| baseline  | 65.00% (39/60) | 89.80% (44/49)  | 8/18           | 0/5            |
| t6_stage2 | 65.00% (39/60) | 91.84% (45/49)  | 6/18           | 0/5            |
```

两个反直觉信号：
1. **Baseline 在 fail 集上 65% 正确** —— 按 scope 定义应该 0%（这些 prompt 就是 baseline 答错的）
2. **Baseline 在 ok 集上 89.8%** —— 按定义应该 ~100%（这些就是 baseline 答对的）
3. T6 跟 baseline 同样 65% → 看起来 SFT 毫无作用

## 根因

**Eval 用的 inference config 跟 scope 生成 config 不一致**。

Scope 定义（`docs/archive/ablation_index.zh.md` § Setting + `scope_fail_prompts.json` 每条 metadata）：
- `temperature = 0`
- `block_length = 32`
- **`gen_length = 128`**
- `num_steps = 128`（coupled）
- `remasking = "low_confidence"`

v1.6.0 的 `v16_eval.py` 默认：
```python
ap.add_argument("--gen_length", type=int, default=192)   # ← 不匹配
ap.add_argument("--block_length", type=int, default=32)
ap.add_argument("--temperature", type=float, default=0.0)
```

`gen_length=192` 比 scope 定义的 128 大得多。从 A6 findings 我们**早已知道** gen_length 变化本身能 rescue 错题：

```
per_length: g64=27  g96=36  g128=42  g160=49  g192=39  g256=40
```

A6 报告 `g=192` 能救 39/60 prompt —— 跟本次 baseline 的 39/60 **完全对上**。
也就是说 "baseline 65%" 这个数字**完全来自 gen_length 从 128 放大到 192 的 A6 effect**，跟 baseline 模型能力本身无关。

T6 训练可能是 work 的，但被 g=192 的 noise 淹没 → 两个 ckpt 看起来都是 65%。

## 次要影响链

因为 eval config 错，更致命的是：**无法判断 T6 SFT 是否 work**。
- 如果改成 g=128 eval，baseline 应 0%，T6 可能 >0%（真实信号）
- 也可能 T6 在 g=128 下仍 0% → 表示 SFT 没迁移到未见过的困难题上
- 不换 config，永远看不出来

## 修复

在 v1.6.1：

1. **`scripts/validate/v16_eval.py`** 默认改回 canonical:
   ```python
   --gen_length  default 128  (was 192)
   --block_length default 32
   --temperature  default 0.0
   ```
   加了注释解释这必须匹配 scope creation。

2. **`scripts/validate/regen_scope.py`（新）**：
   清洁重建 scope —— 在 canonical config 下跑 baseline LLaDA-8B-Instruct 完整 gsm8k test 集（1319 题），重新 partition fail / ok。
   保证未来 eval 跟 scope 完全同 config。

3. **`scripts/run_v1.6.sh`** 加 `EVAL_GEN_LENGTH=128` 独立变量，不 reuse `T7_GEN_LENGTH`。

4. Archive 这个 doc，以后 post-mortem 有据可查。

## 教训

- **Canonical config 要在 scope 文件里 embed**，eval 脚本读 scope 时直接 pull 用，不让用户/脚本默认值"自由发挥"。
  - Scope item 已经有 `num_steps`, `block_length`, `dag_seq_len` 字段 —— eval 应该**读这些字段**作为配置源，而不是从 argparse default
  - 待做 v1.6.2：eval 脚本 auto-detect config from scope file metadata

- **Eval 脚本要跑 sanity check**：加载完 ckpt 先跑一个 known-correct prompt（如 "What is 2+2?"），return 非空 + extract 到数字。没过直接 raise。（v1.6.1 已实现）

- **首条 prompt 的 exception 立刻 raise**，不要 `out=""` 继续。之前 109 条全空是因为 exception 被 except 吞了。（v1.6.1 已实现）

## 相关 commit

- v1.6.0 → v1.6.1 settings fix：待定
- `regen_scope.py`：待定
- v1.6.1 release：待定

## 相关 doc

- [`ablation_index.zh.md` § Setting & Definitions](../ablation_index.zh.md#Setting--Definitions)
- [`finding_gen_length_sensitivity.zh.md`](../finding_gen_length_sensitivity.zh.md) —— A6 per-length table
