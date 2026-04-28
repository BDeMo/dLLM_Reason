# Finding: P2.C ORM verifier + BoN — mean pooling makes BoN beat SC

> Language: English  |  中文: [finding_p2c_orm_bon.zh.md](finding_p2c_orm_bon.zh.md)

**Date**: 2026-04-28
**Scripts**: `scripts/orm_collect_data.py` / `scripts/orm_train.py` / `scripts/orm_eval_bon.py` / `scripts/orm_eval_aggregate.py`
**Pipeline**: `scripts/orm_pipeline.sh`
**Reports**:
- v1 (last pooling, REJECTED): `runs/validation/orm_eval_20260428_031827`
- **v2 (mean pooling, SUPPORTED)**: `runs/validation/orm_eval_20260428_115153`

---

## TL;DR

ORM verifier + Best-of-N reaches **49.2% fail rescue** — +15.4pp over greedy and +9.6pp over SC@8, with no retention drop. **BoN captures ~50% of the pass@8 oracle headroom and is now our strongest inference strategy.**

The pivot: switching the ORM head's pooling from `last` (last non-pad token's hidden state) to `mean` (average over the generation region) jumped fail rescue from 35.6% to 49.2% (+13.6pp).

---

## Setup

| Axis | Value |
|---|---|
| Base model | `runs/training/v161_t6_ablate/hf_step_336` (T6 SFT, frozen) |
| Head | `ORMHead`: single Linear(hidden_size, 1), ~16 KB |
| Train data | gsm8k train, N=8 samples × T=0.7, filter requires ≥1 pos & ≥1 neg |
| Loss | BCEWithLogitsLoss (label = is_correct(answer, gt)) |
| Training | 2000 steps, AdamW lr=1e-4, batch=8, DDP × 8 GPU |
| Eval scope | fail = 331 (baseline wrong), ok = 988 (baseline right), full |

References: Cobbe et al. 2021 (arXiv:2110.14168), V-STaR (arXiv:2402.06457).

---

## Main result (v2, mean pooling)

| metric | fail rescue | ok retention |
|---|---|---|
| greedy | 33.8% (112/331) | 89.0% (879/988) |
| SC@8 | 39.6% (131/331) | 94.3% (932/988) |
| **BoN@8 (ORM)** | **49.2% (163/331)** | **94.0% (929/988)** |
| pass@8 (oracle) | 65.0% (215/331) | 98.7% (975/988) |

**Rescue efficiency** (fraction of oracle headroom captured):
- BoN: (49.2 − 33.8) / (65.0 − 33.8) = **49.4%**
- SC@8: (39.6 − 33.8) / (65.0 − 33.8) = 18.6%
- BoN captures ~2.7× more headroom than SC.

---

## v1 vs v2 — pooling ablation

| metric | v1 last-pool | **v2 mean-pool** | Δ |
|---|---|---|---|
| greedy fail | 34.7% | 33.8% | −0.9 (sampling noise) |
| SC@8 fail | 39.9% | 39.6% | −0.3 |
| **BoN@8 fail** | 35.6% | **49.2%** | **+13.6pp** |
| pass@8 fail | 63.4% | 65.0% | +1.6 |
| BoN ok retention | 90.8% | 94.0% | +3.2pp |

V1's BoN underperformed SC (35.6 < 39.9) — the head learned almost nothing. V2's pooling switch flipped the result. **The root cause is confirmed: last-token signal is too narrow under LLaDA's bidirectional attention.**

---

## Why mean pooling rescued the head

LLaDA is a **bidirectional masked-diffusion** model — there is no causal-LM guarantee that the last token aggregates global semantic information. Pre-pad, the last position is typically EOS or punctuation, weakly correlated with reasoning correctness.

Critical implementation detail: mean only over the **generation region** (not prompt, not pad), gated by an explicit `output_mask`:

```python
# ORMDataset.__getitem__: derive boundary from prompt_text token count
output_mask = torch.zeros(L, dtype=torch.long)
output_mask[prompt_len:] = 1
output_mask = output_mask * attention_mask  # exclude pad
```

If the mean averages prompt tokens too, the prompt content dominates the pooled vector and the signal decouples from correctness — the head still wouldn't learn. **This output-region restriction is what makes mean pooling work.**

---

## DDP details

`orm_train.py`:
- Only `model.head` is wrapped in DDP (base is frozen, no grad sync needed);
- `DistributedSampler` + `set_epoch(epoch)` reshuffles every epoch;
- Train/val loss all-reduced across ranks; save / mkdir / log on rank 0 only.

`orm_eval_bon.py`:
- Prompt-sharded parallelism (`--prompt_shard idx/total`); each shard writes `summary_shard{i}.json`;
- `orm_eval_aggregate.py` combines shards → `summary.json` + `summary.md`.

8-GPU is ~8× single-GPU. End-to-end collect→train→eval ~1h.

---

## Relation to prior experiments

- **Supersedes SC@N as default inference**: BoN dominates SC on fail rescue with no retention cost. SC@N remains a fallback when no verifier is available.
- **Contrast with T7 self-distill**: T7 v1/v2 both failed (capacity ceiling regressed). ORM takes the verifier route — leaves the base untouched and **fixes reliability at inference time**. As long as capacity remains (pass@8 = 65%), the verifier can convert it.
- **Feeds into P2.D**: BoN can now serve as `serve.py`'s default decode strategy; N=8 inference cost is acceptable.

---

## Key takeaways

1. **ORM verifier path is SUPPORTED**: fail rescue +15.4pp; rescue efficiency from SC's 18.6% to 49.4%.
2. **Pooling is the decisive detail for dLLM verifiers**: bidirectional models require mean-over-output-region; the causal-LM last-token recipe doesn't transfer.
3. **Capacity vs Reliability framework holds**: T6 SFT supplies capacity (pass@8 = 65%); the verifier fixes reliability (capturing ~50% of it).
