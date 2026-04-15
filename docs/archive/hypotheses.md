# Post-DAG Pivot — Hypothesis Registry

> Language: English  |  中文: [hypotheses.zh.md](hypotheses.zh.md)

**Context**: `finding_dag_search_zero_rescue.md` shows that v1.5.3's static DAG greedy search has rescue rate = 0 on gsm8k. This document registers the candidate hypotheses for "where is the real bottleneck", each paired with a runnable validation script and explicit evidence thresholds.

**Rules**:
- Validate one hypothesis at a time.
- Each script produces a **structured JSON result** (`runs/validation/h{n}_*.json`) with a top-level `verdict = SUPPORTED | REJECTED | INCONCLUSIVE`.
- Append the verdict to the board at the bottom of this file as soon as a run completes.
- The failing-case scope is fixed at `scope_fail_prompts.json` (produced by `h0_forensics.py`); all subsequent experiments reuse it.

---

## Hypothesis list

### H0 (exploratory): Failure modes are classifiable
**Not a falsifiable hypothesis** — defines a controlled scope for downstream experiments. Reads 137 `correct=0` samples from `episodes.db` and buckets them by error type (early-step commit error / late-stage numeric error / format error / can't solve at all).

**Output**: `runs/validation/scope_fail_prompts.json` with prompt / gt / output / error_category.
**Threshold**: no verdict, artifact only.

---

### H1: Commit-once-never-revise is the main bottleneck
**Claim**: enabling a **revise hook** on fail cases (every few steps, reset committed tokens with conf < τ back to mask) produces rescue_rate significantly > 0.

**Script**: `scripts/validate/h1_remask_rescue.py`
**Method**:
- Pull H0's fail set (up to N=50 prompts for time control).
- Per prompt, run two inferences:
  1. baseline = `llada_generate(T=0, remasking=low_confidence)` (native)
  2. revise = native + every `revise_every=8` steps, reset committed tokens with conf < `revise_thresh=0.3` back to mask
- Compare correctness.

**Verdict thresholds**:
- `rescue_rate = (revise_correct ∧ ¬baseline_correct) / N`
- `rescue_rate ≥ 5%` → **SUPPORTED** (H1 holds)
- `rescue_rate ≤ 1%` → **REJECTED**
- In between → **INCONCLUSIVE**

---

### H2: T=0 + bidirectional attention makes unmask order nearly irrelevant
**Claim**: on a fixed prompt, the output variance from **changing order (DAG)** is much smaller than the variance from **changing sampling content (temperature)**. The order axis carries far less signal than the content axis.

**Script**: `scripts/validate/h2_order_vs_content.py`
**Method**:
- Pull K=20 prompts from the fail set.
- **Content axis**: same scheduler (`low_confidence`, no DAG), T ∈ {0.0, 0.3, 0.7}, 3 samples each → 9 outputs.
- **Order axis**: T=0.0 fixed, 3 schedulers (`low_confidence`, `random_remask`, `cot_dag`) → 3 outputs.
- Per prompt, compute the variance of normalized edit distance across outputs.

**Verdict thresholds**:
- `order_var / content_var < 0.3` → **SUPPORTED** (H2 holds: order signal is weak)
- `order_var / content_var > 0.7` → **REJECTED**
- In between → **INCONCLUSIVE**

---

### H3: LLaDA-instruct hits its capability ceiling on these 137 prompts
**Claim**: even with more sampling diversity (temperature + N resamples), pass@N on these prompts stays ≈ 0.

**Script**: `scripts/validate/h3_passN_at_temperature.py`
**Method**:
- Pull K=30 prompts from the fail set.
- Per prompt × T ∈ {0.3, 0.7, 1.0} × N=8 samples → compute pass@1 / pass@4 / pass@8.
- Control: same K prompts from `init_ok`.

**Verdict thresholds**:
- fail set `pass@8 < 5%` and control `pass@8 > 90%` → **SUPPORTED** (H3 holds: capability ceiling)
- fail set `pass@8 > 20%` → **REJECTED** (capacity not saturated; diversity can rescue)
- In between → **INCONCLUSIVE**

---

### H4 (backup): Block boundaries inject most of the errors
**Claim**: errors cluster in the first k committed tokens of each block (high-confidence trap).
**Script**: placeholder, revisited after H1/H2/H3.

---

## Execution order

1. **H0** (zero cost, run first to generate scope) → `scope_fail_prompts.json`
2. **H1** (cheapest, tells us directly whether the pivot direction is correct)
3. **H3** (if H1 is rejected, H3 disambiguates "order useless but content diversity works" vs "model simply can't do it")
4. **H2** (supplementary evidence, not on the critical path)

---

## Verdict board (auto-updated)

> Overwritten by `scripts/validate/aggregate_verdicts.py` from the latest `runs/validation/h{1,2,3}_*/summary.json`.
> Write manual notes **above** this table; contents here are rewritten each aggregate run.

| Hypothesis | Script | Verdict | Key numbers | Date |
|---|---|---|---|---|
| H0 | `h0_forensics.py` | DONE | 137 fail prompts → runs/validation/scope_fail_prompts.json | 2026-04-15 |
| H1 | `h1_remask_rescue.py` | — | — | — |
| H2 | `h2_order_vs_content.py` | — | — | — |
| H3 | `h3_passN_at_temperature.py` | — | — | — |
