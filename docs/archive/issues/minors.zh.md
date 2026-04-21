# Minor Bugs —— 跨版本小问题记录

> 跨 release 的**小 bug 日志**。重大问题单独建档 in `docs/archive/issues/<name>.zh.md`。
>
> **Append-only**：新 bug 加到顶部（倒序时间），不动旧条目。
>
> 每条格式：`[YYYY-MM-DD] [版本] 标题 — 根因一句 + 修复 commit`

---

## 2026-04 v1.6.x

### [2026-04-20] v1.6.1 — `--batch_size` forward miss on no-trailing-\ line
- `run_t6_shards.sh` 里 aggregate pass 最后一行无 `\` 续行，`sed replace_all` 没匹配到 → 只第一处（shard 调用）有 `--batch_size`，第二处（aggregate）没转发
- Fix: commit after 99778b1, 手动 edit 第二处

### [2026-04-20] v1.6.0 — Qwen3.5 repo 名字无 `-Instruct` 后缀
- 我按 Qwen2.5 命名惯例假设了 `Qwen/Qwen3.5-4B-Instruct`，但 HF 实际 repo 叫 `Qwen/Qwen3.5-4B`（chat 是默认，base 才加 `-Base`）
- Fix: commit `3cbd84e`

### [2026-04-19] v1.5.x/v1.6 — Bash `$GROUPS` readonly array collision
- `run_ss_shards.sh` 用 shell 变量 `GROUPS` 存 fail/ok groups，但 `$GROUPS` 是 bash 内置的只读数组（存当前用户的 group IDs）。root 用户下 `$GROUPS[0] = 0` 覆盖掉赋值，导致 orchestrator 拿 `groups="0"` 过去，scope 解析 0 条
- Fix: rename `GROUPS → PROMPT_GROUPS`，commit `<...>` (ss shards era)

### [2026-04-17] v1.5.x — Python stdout full-buffer → shard log 显示空白
- Python `print()` 重定向到文件时默认 full-buffered（4-8 KB），`pp.tick()` 每 prompt 一次、几十字节，半小时都 flush 不了
- 看起来像 shard 崩了，实际在跑
- Fix: `PYTHONUNBUFFERED=1 python -u`，commit `3ac3ac1`

### [2026-04-19] v1.6.0 — `wait ""` on empty SHARD_PIDS array
- `"${SHARD_PIDS[@]:-}"` 空数组展开成单个空字符串，`wait ""` 报 "not a pid or valid job spec"
- Fix: `if [[ ${#SHARD_PIDS[@]} -gt 0 ]]` guard，commit 在 ss shards era

### [2026-04-19] v1.6.0 — `build_jsonl_dataset` 1-record split 边界
- `max(1, int(1 * 0.1)) = 1` → val 拿走唯一记录，train 空
- 用户 dry-run 报 `0 train / 1 val`，tokenizer 能 load 但训不起来
- Fix: `n_total < 2` 时 skip split + clamp `n_val ≤ n_total-1`，commit `b62c0d4`

### [2026-04-20] v1.6.0 — `t6t7_train.py` HF export 用错 attribute 名
- 我假设 `LLaDAWrapper` 暴露 `._model` / `.model` / `.model_internal` 任一个
- 实际是 `self._llada`（见 `src/dllm_reason/models/llada.py:109`）
- Fix: commit `7f5b5cf`

### [2026-04-20] v1.6.0 — `requires_grad=False` 默认 after `from_pretrained`
- `AutoModel.from_pretrained(..., device_map="auto", torch_dtype=bfloat16)` 某些路径加载后参数 `requires_grad=False`
- Finetuner 只 `.train()` 没改 grad flag → loss 无 `grad_fn` → backward 崩
- Fix: 显式 `for p in model.parameters(): p.requires_grad_(True)`，commit `c8b17d7`

---

## 格式与规则

- **Commit hash** 空着也行（历史 bug 补记不强求找 hash）
- **"Fix:"** 那行如果是多步修复，写主 commit + "see also ..."
- **症状**一句，**根因**一句，**修复**一句 —— 不超过 3 行
- 复杂的放 `docs/archive/issues/<name>.zh.md`，这里只留一个指针
