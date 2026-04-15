# Hypothesis Validation Suite

对应 `docs/archive/hypotheses.md`。v1.5.3 DAG search 失效后的 pivot 验证脚本，一次跑一条假设。

## 执行顺序

```bash
# 1) 零成本，生成所有后续实验的 scope（137 条 init_fail）
python scripts/validate/h0_forensics.py

# 2) 最便宜的因果验证 — revise hook 能不能救回 fail 案例
python scripts/validate/h1_remask_rescue.py --n 50

# 3) 如果 H1 证据不够，用 H3 区分"采样不够多样"vs"真不会"
python scripts/validate/h3_passN_at_temperature.py --n_fail 30 --n_ok 30

# 4) 补充证据 — order 轴方差 vs content 轴方差
python scripts/validate/h2_order_vs_content.py --n 20
```

## 产出

所有结果写入 `runs/validation/`：
- `scope_fail_prompts.json`（H0 生成，后续复用）
- `h1_remask_rescue_YYYYMMDD_HHMMSS.json`
- `h2_order_vs_content_YYYYMMDD_HHMMSS.json`
- `h3_passN_YYYYMMDD_HHMMSS.json`

每个 JSON 顶层含 `verdict = SUPPORTED | REJECTED | INCONCLUSIVE` 以及 `config` / 关键数字 / per-prompt 明细。

## 跑完后

在 `docs/archive/hypotheses.md` 底部"结论板"追加一行，记录 verdict + 关键数字 + 产出文件路径。
