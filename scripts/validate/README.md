# Hypothesis Validation Suite

对应 `docs/archive/hypotheses.md`。v1.5.3 DAG search 失效后的 pivot 验证脚本。

## Run 目录约定（每个 hypothesis 一个 run_dir）

```
runs/validation/
├── scope_fail_prompts.json           # H0 产出：init_fail (correct=0)，H1/H2/H3 复用
├── scope_ok_prompts.json             # H0 产出：init_ok   (correct=1)，H3 对照组
├── h1_remask_<ts>/
│   ├── config.json                   # CLI + hypothesis + timestamp
│   ├── per_prompt/
│   │   ├── 0000.json                 # 单条 prompt 结果
│   │   └── ...
│   ├── progress.jsonl                # append-only 进度日志
│   └── summary.json                  # 聚合后 verdict
├── h2_order_content_<ts>/   (同结构)
└── h3_passN_<ts>/           (同结构；per_prompt 文件名为 {fail|ok}_XXXX.json)
```

**增量 + resume**：每跑一条 prompt 立刻原子写 `per_prompt/XXXX.json`。中途 Ctrl-C / OOM / 掉线后：
```bash
python scripts/validate/h1_remask_rescue.py --n 137 --resume \
    --run_dir runs/validation/h1_remask_20260415_083000
```
会跳过已完成的 idx，只补未完成的。

## 本地开发 / dry-run

```bash
# H0 不需 GPU，本地直接跑
python scripts/validate/h0_forensics.py

# h1/h2/h3 带 --dry_run：不加载 torch，只打印会跑什么 + 建好 run_dir
python scripts/validate/h1_remask_rescue.py --n 2 --dry_run
python scripts/validate/h2_order_vs_content.py --n 2 --dry_run
python scripts/validate/h3_passN_at_temperature.py --n_fail 2 --n_ok 2 --dry_run
```

## 服务器 GPU 执行

```bash
cd dLLM_Reason && git pull origin dev

# 一键全跑（顺序：H0 → H1 → H3 → H2 → aggregate）
bash scripts/validate/run_all.sh

# 或手动分步
python scripts/validate/h0_forensics.py
python scripts/validate/h1_remask_rescue.py --n 137
python scripts/validate/h3_passN_at_temperature.py --n_fail 30 --n_ok 30 --n_samples 8
python scripts/validate/h2_order_vs_content.py --n 20
python scripts/validate/aggregate_verdicts.py

# 打包回传
tar czf validation_results.tar.gz runs/validation/ docs/archive/hypotheses.md
```

**GPU / 时间预估**（llada-instruct, H100 / A100）：
| 脚本 | N | 次数 | 单次 | 估时 |
|---|---|---|---|---|
| H1 | 137 | 2 gen/prompt = 274 | ~5s | ~25min |
| H3 | 60 | 3T × 8 = 24/prompt = 1440 | ~5s | ~2h |
| H2 | 20 | 12/prompt = 240 | ~5s | ~20min |

Disk：`runs/validation/` 约 20 MB。

## 本地聚合 review

```bash
tar xzf validation_results.tar.gz

# 把 3 个 summary.json 填回 hypotheses.md 结论板
python scripts/validate/aggregate_verdicts.py

# 看结果
tail -20 docs/archive/hypotheses.md
```

## Verdict 判据（同 `hypotheses.md`）

| 假设 | SUPPORTED | REJECTED |
|---|---|---|
| H1 rescue_rate | ≥ 5% | ≤ 1% |
| H2 order_var / content_var | < 0.3 | > 0.7 |
| H3 fail_pass@8 | < 5% 且 ok_pass@8 > 90% | > 20% |

## 执行决策树

```
H1 ──┬─ SUPPORTED (rescue ≥ 5%)   → 升 D/F/H, 降 G/E, sampler 接 correction_head
     ├─ REJECTED  (rescue ≤ 1%)   → 看 H3
     │                               ├─ SUPPORTED → 训练端 pivot (A/B/#11)
     │                               └─ REJECTED  → H2 兜底 (B/F diversity)
     └─ INCONCLUSIVE              → H2 + H3 同时上补充证据
```
