# Plan: Distillation 4 决策（a/iv/α/B）

**日期**：2026-04-16
**状态**：active（pending actual SS data to validate）
**前置上下文**：[2026-04-16_strategy_search_design.zh.md](2026-04-16_strategy_search_design.zh.md) —— SS 完会产 `winners.json`，需要定义"怎么喂 SFT"
**继任**：—

## 目标

把 SS 的 `winners.json` 转成 SFT JSONL，让下一阶段的 policy head 学 `prompt → strategy` 映射。需要 4 个设计决策。

## 决策

**最终选型**：`a / iv / α / B`（β 作为备用 flag）

### 决策 1 —— Winner 选什么

**选 (a) cheapest 单头** —— pass@1 ≥ 1 的配置里，`gen_length × num_samples` 最小的。
- 理由：部署最便宜（单配置推理）；数据集最全（T=0 + T>0 都能做 cheapest）
- 备选 `shortest / most_reliable / deterministic`（`ss_to_sft.py` 里都生成了，但 SFT 先只用 cheapest）

### 决策 2 —— Output 序列化格式

**选 (iv) key=value 紧凑文本**：
```
bl=32 tmpl=answer_marker pos=prefix gen=128 T=0
```
- 理由：~10 tokens；不用改 tokenizer；一行 regex 反向 parse；不 key-order-敏感
- `num_samples` 不放进 target（T→N 一一对应，模型不用学）
- temperature 用 `:g` 格式（T=0 不是 T=0.0）
- 抛弃：(i) JSON string 太长；(ii) NL 有歧义；(iii) special tokens 要改 vocab

### 决策 3 —— Input 格式

**选 (α) 裸 prompt**：
```
Q: {prompt}\nStrategy: 
```
- 先最简 baseline。如果学得差再上 (β) K-shot demo
- `ss_to_sft.py --k_shot N` 作为 β backup 已实现：demos 从 train split 采（避免 val leakage），demos 排除 abstain 例子

### 决策 4 —— 没 winner 的 prompt 怎么办

**选 (B) abstain 标签** —— `<UNSALVAGEABLE>`
- 理由：教 policy 学会"refuse"能力 → 下游可接 verifier / tool / human
- 抛弃：(A) 跳过不喂；(C) fallback 到 baseline（会教模型一个已知错的策略）

## 预期输出

- `<ss_run_dir>/sft/sft_train.jsonl`
- `<ss_run_dir>/sft/sft_val.jsonl`（val_frac=0.1, stratified per group）
- `<ss_run_dir>/sft/sft_stats.json`（abstain 率、top-5 strategy 分布）
- `<ss_run_dir>/sft/sft_manifest.json`（决策 + seed + format spec，可复现）

## 下一步（等 SS 跑完）

1. `ss_analyze.py --run_dir <ss_run_dir>` → 看 oracle rate 和 template_position novelty
2. `ss_to_sft.py --run_dir <ss_run_dir>` → 产 SFT JSONL
3. Pick base model（LLaDA-Instruct？或 Qwen/Llama 小模型）做 policy head
4. Train baseline SFT，跟 `uniform-best-strategy` (比如 g=160/bl=32/answer_marker/prefix/T=0) 比
5. 定义 success metric：`policy pass@1 / oracle pass@1`

## Updates

- **2026-04-16**: 决策敲定，`ss_to_sft.py` 已实现 commit `cba1bad`
- **2026-04-19**: SS 跑挂，decisions 仍 active，**等重新规划后**拿到 winners 再 apply

## Retrospective（pending）

`ss_to_sft.py` 的实际行为要跑起来才知道是否合理。具体 TBD 项：
- SFT 训练集大小够不够（预期 ~100 条 non-abstain），可能要扩 ok 组或做 data augmentation
- `<UNSALVAGEABLE>` 的 abstain 实测占比（ceiling 5 + 若干边缘 prompt）
- 如果 cheapest winner 集中在 1-2 个 strategy（entropy 低），SFT 可能退化成学 majority 策略，不会真 per-prompt —— 这时需要换 target 或加 contrastive loss
