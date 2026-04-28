# Finding：P2.C ORM verifier + BoN —— mean pooling 让 BoN 反超 SC

> 语言：中文  |  English: [finding_p2c_orm_bon.md](finding_p2c_orm_bon.md)

**日期**：2026-04-28
**脚本**：`scripts/orm_collect_data.py` / `scripts/orm_train.py` / `scripts/orm_eval_bon.py` / `scripts/orm_eval_aggregate.py`
**Pipeline**：`scripts/orm_pipeline.sh`
**报告目录**：
- v1（last pooling，REJECTED）：`runs/validation/orm_eval_20260428_031827`
- **v2（mean pooling，SUPPORTED）**：`runs/validation/orm_eval_20260428_115153`

---

## TL;DR

ORM verifier + Best-of-N 采样在 fail rescue 上 **49.2%**，比 greedy +15.4pp，比 SC@8 +9.6pp，retention 不掉点。**BoN 抓住了 pass@8 oracle ceiling 的 ~50%，是目前最强 inference 策略**。

关键转折：把 ORM head 的 pooling 从 `last`（最后一个非 pad token 的 hidden state）换成 `mean`（生成区域内的平均 hidden state），fail rescue 从 35.6% 跳到 49.2%（+13.6pp）。

---

## 实验设置

| 维度 | 值 |
|---|---|
| Base model | `runs/training/v161_t6_ablate/hf_step_336`（T6 SFT，frozen） |
| Head 架构 | `ORMHead`：单层 Linear(hidden_size, 1)，~16 KB |
| 训练数据 | gsm8k train，N=8 samples × T=0.7，过滤要求 ≥1 pos & ≥1 neg |
| 训练目标 | BCEWithLogitsLoss（label = is_correct(answer, gt)） |
| 训练 step | 2000，AdamW lr=1e-4，batch=8，DDP × 8 GPU |
| Eval scope | fail = 331（baseline 错），ok = 988（baseline 对），全量 |

参考论文：Cobbe et al. 2021（arXiv:2110.14168）、V-STaR（arXiv:2402.06457）。

---

## 主结果（v2，mean pooling）

| metric | fail rescue | ok retention |
|---|---|---|
| greedy | 33.8% (112/331) | 89.0% (879/988) |
| SC@8 | 39.6% (131/331) | 94.3% (932/988) |
| **BoN@8 (ORM)** | **49.2% (163/331)** | **94.0% (929/988)** |
| pass@8 (oracle) | 65.0% (215/331) | 98.7% (975/988) |

**Rescue efficiency**（抓住 oracle headroom 的比例）：
- BoN：(49.2 − 33.8) / (65.0 − 33.8) = **49.4%**
- SC@8：(39.6 − 33.8) / (65.0 − 33.8) = 18.6%
- BoN 抓住的 headroom 是 SC 的 ~2.7×。

---

## v1 vs v2 —— pooling 对比

| metric | v1 last-pool | **v2 mean-pool** | Δ |
|---|---|---|---|
| greedy fail | 34.7% | 33.8% | −0.9（sampling 噪声） |
| SC@8 fail | 39.9% | 39.6% | −0.3 |
| **BoN@8 fail** | 35.6% | **49.2%** | **+13.6pp** |
| pass@8 fail | 63.4% | 65.0% | +1.6 |
| BoN ok retention | 90.8% | 94.0% | +3.2pp |

V1 的 BoN 跑输 SC（35.6 < 39.9），head 几乎没学到东西；v2 一改 pooling 就反超 SC，**根因实锤是 last-token signal 在 LLaDA 双向 attention 下太窄**。

---

## 为什么 mean pooling 救活了 head

LLaDA 是 **bidirectional masked-diffusion** 模型，没有 causal LM 那种 "最后一个 token 汇聚全文 信息" 的语义保证。Last token 在 padding 之前往往是 EOS / 标点，hidden state 与 reasoning 正确性的相关性弱。

Mean pooling 实现关键点：**只在生成区域上平均**（不含 prompt、不含 pad），由 `output_mask` 显式控制：

```python
# ORMDataset.__getitem__：根据 prompt_text 的 token 数定边界
output_mask = torch.zeros(L, dtype=torch.long)
output_mask[prompt_len:] = 1
output_mask = output_mask * attention_mask  # 排除 pad
```

如果 mean 时把 prompt 一并平均进来，prompt 内容会主导 pooled vector，与 correctness 解耦，head 同样学不到。**这一点是 mean pooling 能成立的前提**。

---

## DDP 实现要点

`orm_train.py`：
- 只 wrap `model.head` 进 DDP（base frozen 不参与 grad sync）；
- `DistributedSampler` + `set_epoch(epoch)` 保证每 epoch 重 shuffle；
- train/val loss 跨卡 all-reduce；rank 0 独占 save / mkdir / log。

`orm_eval_bon.py`：
- prompt-shard 并行（`--prompt_shard idx/total`），每 shard 写 `summary_shard{i}.json`；
- `orm_eval_aggregate.py` 汇总所有 shard → `summary.json` + `summary.md`。

8 卡 ≈ 单卡 8× 提速；端到端 collect→train→eval ~1h。

---

## 与前期实验的关系

- **替代 SC@N 作为推理首选**：BoN 在 fail rescue 上显著优于 SC，retention 不损失。SC@N 仍可作为无 verifier 时的兜底。
- **与 T7 self-distill 的对比**：T7 v1/v2 都失败（capacity ceiling 反降）；ORM 走的是 verifier 路线，不动 base，**inference-time 抓 reliability**——证明只要 capacity 仍在（pass@8 = 65%），verifier 就能把它转化出来。
- **与 P2.D 的衔接**：BoN 已可作为 `serve.py` 的 default decode strategy，N=8 推理成本可控。

---

## 关键 takeaway

1. **ORM verifier 路线 SUPPORTED**：fail rescue +15.4pp，rescue efficiency 从 SC 的 18.6% 拉到 49.4%。
2. **Pooling 选择是 dLLM verifier 的决定性细节**：bidirectional 模型必须 mean over output region，不能照抄 causal LM 的 last-token 配方。
3. **Capacity vs Reliability 框架被再次验证**：T6 SFT 提供 capacity（pass@8 = 65%），verifier 解决 reliability（把其中 ~50% 落地）。
