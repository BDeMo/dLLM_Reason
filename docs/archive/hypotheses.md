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
- **Order axis**: T=0.0 fixed, 3 `block_length` values {16, 32, 64} (same `low_confidence` scheduler; changing block size changes the commit order across steps) → 3 outputs.
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

### A3: Span-level revise beats single-token revise
**Claim**: errors live in **contiguous spans**, not individual tokens. A revise hook that masks an entire window whose **mean committed confidence** < τ rescues fail cases that H1's single-token τ missed.

**Script**: `scripts/validate/a3_span_revise.py`
**Method**: sliding window (default `window_size=4, revise_thresh=0.4, revise_every=8`) over committed tokens via `F.conv1d`; mask all committed positions in bad windows.
**Verdict thresholds**: same as H1.

### A4: Block layout matters — some split rescues fail cases
**Claim**: the default `block_length=32` is sub-optimal for some reasoning structures; at least one layout in `{8, 16, 32, 64, short_then_long}` rescues fails.

**Script**: `scripts/validate/a4_block_rerank.py`
**Method**: per prompt, run with block_length ∈ {8, 16, 32, 64} and one non-uniform layout (first 64 tokens at block=16, last 64 at block=64). Record `any_layout_correct`.
**Verdict thresholds**: `any_layout_rescue_rate ≥ 5%` SUPPORTED / ≤ 1% REJECTED.

### A5: Prompt template ensemble rescues fail cases
**Claim**: a CoT-inducing or answer-prefix template shifts the output distribution enough to rescue.

**Script**: `scripts/validate/a5_prompt_template.py`
**Method**: per prompt, append one of {baseline / "\nLet's solve this step by step." / "\nStep 1:" / "\nAnswer:"}. Record `any_template_correct`.
**Verdict thresholds**: same as A4.

---

### E1: Does A6's g160 gain come from "space" or "extra compute steps"?
**Context**: A6 `gen_length=160` rescues 20% of fail cases, but A6 couples `num_steps` and `gen_length` (both track the g value). If the rescue comes from extra compute steps, it's evidence for latent reasoning; if it only comes from extra token positions, A6 is just budget calibration and says nothing about latent reasoning.

**Script**: `scripts/validate/e1_gen_vs_steps.py`
**Method**: run 3 configs on the same 60 fail prompts:
1. **C_g128_s128** (baseline)
2. **A_g160_s160** (= A6 g160, space↑ + steps↑)
3. **B_g128_s160** (space locked at 128, only num_steps bumped to 160)

If A rescue > 0 but B rescue ≈ 0 → **space-effect** (latent reasoning REJECTED).
If A ≈ B rescue > 0 → **compute-effect** (latent reasoning SUPPORTED).

**Verdict thresholds**:
- `rescue_rate_stepsB ≥ 5%` → **SUPPORTED** (latent reasoning holds)
- `rescue_rate_stepsB ≤ 1%` → **REJECTED** (latent reasoning ruled out)
- in between → **INCONCLUSIVE**

### E5: Is A6's g128 tail physically truncated by the token budget?
**Context**: the most trivial counter-explanation — g128 literally runs out of positions before the answer is written, and g160 just adds enough room to finish. No semantic interpretation required.

**Script**: `scripts/validate/e5_truncation_check.py`
**Method**: offline analysis of A6's `tails` field (per prompt × per gen_length, `out[-200:]` stored per run). Heuristics: answer-marker regexes / sentence-end / digit-end / mid-word-end → verdict ∈ {complete, truncated, maybe_truncated, ambiguous}.
**Focus**: g128 tail of the 3 A6-only rescue prompts {0, 19, 51}.

**Verdict thresholds**:
- ≥2/3 truncated-or-maybe → **TRIVIAL_TRUNCATION** (A6 gain is a budget effect)
- ≤1/3 → **NOT_TRUNCATION** (gain can't be explained by truncation)

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
| H0 | `h0_forensics.py` | DONE | 60 fail prompts → runs/validation/scope_fail_prompts.json | 2026-04-16 |
| H1 | `h1_remask_rescue.py` | REJECTED | N=137  base=0  revise=0  rescued=0  broken=0  rescue_rate=0.00% | 2026-04-16 |
| H2 | `h2_order_vs_content.py` | REJECTED | N=20  content_var=0.256  order_var=0.176  ratio=0.754 | 2026-04-16 |
| H3 | `h3_passN_at_temperature.py` | REJECTED | n_fail=60  n_ok=30  fail_p@8=86.67%  ok_p@8=100.00% | 2026-04-16 |
| A3 | `a3_span_revise.py` | REJECTED | N=60  base=42  revise=42  rescued=0  broken=0  rescue_rate=0.00% | 2026-04-16 |
| A4 | `a4_block_rerank.py` | SUPPORTED | N=60  base(bl32)=42  any=47  rescue_rate=8.33%  [bl8=43 bl16=41 bl32=42 bl64=37 short_then_long=37] | 2026-04-16 |
| A5 | `a5_prompt_template.py` | SUPPORTED | N=60  base=42  any=50  rescue_rate=13.33%  [baseline=42 cot_plain=35 cot_step=30 answer=45] | 2026-04-16 |
| A6 | `a6_gen_length.py` | SUPPORTED | N=60  base(g128)=42  any=54  rescue_rate=20.00%  [g64=27 g96=36 g128=42 g160=49 g192=39 g256=40] | 2026-04-16 |
| A4x5 | `a4x5_joint.py` | — | N=60  base=42  joint_any=52  rescue_rate=16.67%  [bl8_baseline=43 bl8_answer=41 bl32_baseline=42 bl32_answer=45 bl64_baseline=37 bl64_answer=40] | — |
| E1 | `e1_gen_vs_steps.py` | REJECTED | N=60  C_g128_s128=42  A_g160_s160=49  B_g128_s160=42  rescue_longA=15.00%  rescue_stepsB=0.00%  a6_only_longA=1/3  a6_only_stepsB=0/3 | 2026-04-16 |
| E5 | `e5_truncation_check.py` | NOT_TRUNCATION | a6_only_g128_trunc=1/3  [idx0:mayb/✗  idx19:comp/✗  idx51:comp/✗] | 2026-04-16 |
