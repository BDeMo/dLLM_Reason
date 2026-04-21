# Minor Bugs —— 跨版本小问题记录

> 跨 release 的**小 bug 日志**。重大问题单独建档 in `docs/archive/issues/<name>.zh.md`。
>
> **Append-only**：新 bug 加到顶部（倒序时间），不动旧条目。
>
> 每条格式：`[YYYY-MM-DD] [版本] 标题 — 根因一句 + 修复 commit`

---

## 2026-04 v1.6.x

### [2026-04-21] v1.6.1 — `regen_scope.py` shard race on per_prompt key
- `prompt_key(i)` 用的是 `enumerate(prompts)` 的本地 i，而 `prompts` 是分片切完的子集 → 8 个 shard **全部从 `t0000` 开始**
- 所有 shard 写 `per_prompt/t0000.json.tmp` 互抢 → 一个 `os.replace(tmp, .json)` 后另一个找不到 tmp → `FileNotFoundError: ... t0000.json.tmp`
- 数据互相覆盖；最终 `scope_fail/ok` 缺失大半
- Fix: `prompt_key` 改用 `rec["source_idx"]`（全局 gsm8k test 索引），所有 shard 写不同 key；aggregate 也按 source_idx 聚合

### [2026-04-21] v1.6.1 — `run_regen_scope_shards.sh` prompt counter 用 `load_dataset` 触发 HF
- bash 启动时 `TOTAL=$(python heredoc { load_dataset("openai/gsm8k") })` 即使本地有 `datasets/gsm8k/test/dataset_info.json` 也会先 ping HF 验 metadata
- User: "为什么要访问 huggingface.co" + "datasets/gsm8k/test/ 存在啊" + "dataset_info.json 也存在啊"
- Fix: bash heredoc 改用 `load_from_disk(str(local_dir))`（不联网）+ parquet fallback；并 `export HF_HUB_OFFLINE=1` `HF_DATASETS_OFFLINE=1` 给整个 shell。同时 `regen_scope.py` 也设这些 env var 防 datasets 库内部 metadata refresh

### [2026-04-21] v1.6.1 — `regen_scope.py` 在 local 缺失时悄悄 fall-back HF
- `resolve_dataset()` 设计是"local 优先，没有就 HF 下载"。在 *单独*跑 `run_regen_scope_shards.sh`（不经 Phase 0）时 `datasets/gsm8k/test/` 不存在 → fallback 真的去 HF → 网络断 / mirror 挂 → "huggingface.co Network unreachable" 错
- User 困惑 "为什么要访问 huggingface.co"
- Fix: `regen_scope.py` + `run_regen_scope_shards.sh` 都加 pre-flight check：local missing 直接 fail 给出 fix 命令（`python scripts/download_datasets.py --datasets gsm8k` 或 `bash run_all_v1.6.1.sh --from_phase 0 --to_phase 0`），**不再悄悄走网络**

### [2026-04-21] v1.6.1 — pipeline 不用本地 `checkpoints/` + `datasets/` 注册路径
- Project 早有 `src/dllm_reason/utils/local_resolve.py` + `resource_registry.py`，能解析 `GSAI-ML/LLaDA-8B-Instruct` → `checkpoints/llada-instruct/` + `openai/gsm8k` → `datasets/gsm8k/<split>/`
- 也有 `scripts/download_models.py` / `scripts/download_datasets.py` 把数据下到注册路径
- v1.6.1 早期版本绕开了这套，直接 `load_dataset(...)` + 加 `--offline` workaround；user 指出方向错
- **正解**：Phase 0 用 `download_models.py` + `download_datasets.py` 把 LLaDA + GSM8K 下到 `checkpoints/llada-instruct/` 和 `datasets/gsm8k/{train,test}/`；后续 phase 通过 `resolve_dataset` 自动命中本地，**`--offline` 不需要**（仍保留作 escape hatch）
- 教训：找现有 infra 之前别先写 workaround

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

## Sharded-write 写法 checklist（防 race）

写新 sharded 脚本前，对照以下检查避免 v1.6.1 出现过的 t0000 race：

1. **每个 shard 写不同 path** —— per-prompt key 必须用**全局唯一**索引（gsm8k 用 `source_idx`，scope_fail/ok 用 `(group, idx)` 中的 `idx` 必须是全局的，不是 enumerate(slice) 局部的）
2. **`enumerate(prompts)` 后才 slice** —— 永远先 enumerate 完整列表，后切片，让原始 index 跟着 tuple 走
3. **聚合时按全局 key 找文件** —— iterate `prompts_full` 而不是 `prompts`，否则 missing 半数据
4. **`load_dataset` 不在 bash heredoc 里调** —— 永远用 `load_from_disk` / parquet / json，HF 网络永不在 hot path
5. **export `HF_HUB_OFFLINE=1` HF_DATASETS_OFFLINE=1` TRANSFORMERS_OFFLINE=1`** —— 任何 sharded scope/data-prep 脚本顶部统一设
6. **`os.replace(tmp, path)` 前确认 tmp 存在** —— 写 race 之外，磁盘满 / 权限错也会触发；至少 try/except 加 print
7. **空数组 wait 守卫** —— `if [[ ${#PIDS[@]} -eq 0 ]]; then ... fi` 防 `wait ""` 报 not-a-pid

## 格式与规则

- **Commit hash** 空着也行（历史 bug 补记不强求找 hash）
- **"Fix:"** 那行如果是多步修复，写主 commit + "see also ..."
- **症状**一句，**根因**一句，**修复**一句 —— 不超过 3 行
- 复杂的放 `docs/archive/issues/<name>.zh.md`，这里只留一个指针
