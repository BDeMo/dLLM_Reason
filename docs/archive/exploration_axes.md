# Post-DAG Exploration Axes

> Language: English  |  中文: [exploration_axes.zh.md](exploration_axes.zh.md)

**Context**: after DAG search was confirmed dead on three independent implementations (greedy / NAS supernet / E2E differentiable — see `finding_dag_search_zero_rescue.md`) and the first batch of hypotheses (H1 token-revise, H2 order-vs-content) came back REJECTED, we need a map of where to look next.

This document indexes the exploration directions as two orthogonal axes:
- **A axis — granularity ladder**: sweep from fine (edge) to coarse (prompt template), changing only what unit of "intervention" is applied at inference time.
- **B axis — orthogonal directions**: changes outside the inference-time sampler (training, tooling, verifier).

**Convention**: each item has a status tag — `DEAD` (disproven), `PLANNED` (script not yet run), `RUNNING` (in progress), `DONE` (verdict recorded in `hypotheses.md`). Move to `DEAD` only after at least one experiment with N ≥ 30 and a recorded verdict.

---

## A axis — granularity ladder

Sweep order: fine → coarse. A1/A2/A3 are `DEAD/REJECTED`; A4 is `SUPPORTED` (block-layout 8.33% rescue, N=60); A5 is `SUPPORTED` (prompt-template 13.33% rescue, N=60); A6 is `SUPPORTED` (gen-length 20.00% rescue, N=60, strongest single-axis signal). Signal appears at block granularity and keeps growing up to the prompt/gen-length level. A4x5 joint 6-cell run perfectly validates overlap prediction (rescue=10=10).

### A1 — edge-level DAG rewiring · **DEAD**
Change which token is unmasked next, edge by edge, on a DAG over positions.
- Evidence: greedy ±1 edge (1319 prompts, 0 rescue, ~3072 edges) + NAS supernet (200/0, 0 edges selected) + E2E differentiable (106/0, 0 edges selected). Three independent optimizers, all 0 rescue.
- Conclusion: at T=0 + bidirectional attention, edge-level ordering carries ≈ 0 signal over the greedy low-confidence schedule.

### A2 — single-token revise hook · **DEAD**
Every few steps, reset committed tokens with conf < τ back to mask.
- Evidence: H1 on 137 fail prompts, rescue_rate = 0 (122/137 never even triggered the hook because committed conf ≥ 0.3 everywhere).
- Conclusion: per-token confidence is not a reliable error signal on fail prompts — confident-but-wrong is the norm.

### A3 — span-level revise · **REJECTED**
Hypothesis: errors live in **contiguous spans** (e.g. a mis-computed arithmetic sub-expression) where any single token may look confident but the **window mean confidence** drops.
- Script: `scripts/validate/a3_span_revise.py`
- Method: sliding window (default `window_size=4`) over committed tokens, compute mean conf via `F.conv1d`, mask ALL tokens in any window whose mean < τ (default `0.4`). Compare to H1's single-token hook.
- Result (N=60): base=42, revise=42, rescued=0, broken=0, `rescue_rate=0.00%` → **REJECTED**.
- Takeaway: combined with H1 (token-level revise REJECTED), the confidence signal carries no actionable error info on fail prompts at any granularity below block.

### A4 — block-layout rerank (absorbs old A6) · **SUPPORTED**
Hypothesis: the fixed `block_length=32` layout is sub-optimal for some reasoning structures; a different split (shorter blocks in the reasoning section, longer in the answer, or simply a different uniform size) rescues errors.
- Script: `scripts/validate/a4_block_rerank.py`
- Method: per fail prompt, sample with block_length ∈ {8, 16, 32, 64} + one non-uniform layout (short-then-long, mimicking "steps → final answer"). Compute `any_layout_correct`.
- Result (N=60): base(bl32)=42, any_layout=47, rescued=5, `rescue_rate=8.33%` → **SUPPORTED**.
- Takeaway: the "order" signal H2 half-hinted at lives at **block boundaries**, not at edges (A1 DEAD) or tokens (A2/A3 DEAD). Block-layout is the first axis with real rescue signal.

### A5 — prompt-template rerank · **SUPPORTED**
Hypothesis: the gsm8k prompt shape constrains the generation too much; CoT-inducing prefixes (or an explicit answer-only prefix) shift the output distribution enough to rescue.
- Script: `scripts/validate/a5_prompt_template.py`
- Method: per fail prompt, run with 4 templates (baseline / "\nLet's solve this step by step." / "\nStep 1:" / "\nAnswer:"). Compute `any_template_correct`.
- Result (N=60): base=42, any_template=50, rescued=8, broken=0, `rescue_rate=13.33%` → **SUPPORTED**.
  - per-template: `baseline=42, cot_plain=35, cot_step=30, answer=45`.
- Takeaway: **counter-intuitive** — the single best template is `answer` (+3 over baseline), while both CoT-inducing templates *hurt* (`cot_plain` −7, `cot_step` −12). The rescue signal is from *template diversity*, not from CoT. LLaDA-instruct seems primed to answer directly on gsm8k, and forcing step-by-step CoT at the prompt level disrupts more prompts than it helps.

### A6 — gen-length rerank · **SUPPORTED**
Hypothesis: the default `gen_length=128` is not optimal for all prompts; different generation lengths can rescue some errors.
- Script: `scripts/validate/a6_gen_length.py`
- Method: per fail prompt, sample with gen_length ∈ {64, 96, 128, 160, 192, 256} (fixed block_length=32). Compute `any_length_correct`.
- Result (N=60): base=42, any_length=54, rescued=12, `rescue_rate=20.00%` → **SUPPORTED**. **Strongest single-axis signal on the A axis.**
  - per-length: `g64=27, g96=36, g128=42, g160=49, g192=39, g256=40`.
  - Sweet spot: g160=49/60=81.7% vs baseline g128=42/60=70%.
- Takeaway: gen_length is the A axis's most powerful knob. g160 alone exceeds A5's any-template ensemble. But g64/g96 are clearly worse, so it's not "longer is better" — there's an optimal length.

### A4×A5 joint 6-cell validation

`{baseline, answer} × {bl8, bl32, bl64}` 6-cell configuration run:
- N=60, base=42, any=52, rescued=10, rescue_rate=**16.67%**
- **Perfectly validates overlap prediction**: predicted 10 rescues, got 10 rescues, 100% match, zero surprises.
- per-cell: `bl8_baseline=43, bl8_answer=41, bl32_baseline=42, bl32_answer=45, bl64_baseline=37, bl64_answer=40`

---

## B axis — orthogonal directions

A axis sweep complete; B axis assessment updated.

### B1 — pass@N diversity sampling · **SUPPORTED (= H3)**
H3 (`h3_passN_at_temperature.py`) is complete.

**Final result (N=30 fail)**: 7/30 rescue (idx=0,2,8,13,15,24,28), rescue_rate=**23.33%** → **SUPPORTED** (overturns earlier "likely INCONCLUSIVE" prediction). 4 prompts stuck (pass@8=0 at all temps).

H3 SUPPORTED means the capacity ceiling hypothesis is **REJECTED** — the model is not fundamentally incapable on these prompts, but rather the default configuration fails. Diversity sampling itself is a valid lever (23.33%), though cheaper A-axis knobs (template/layout/gen_length) achieve comparable or better results.

### B2 — training-side pivot
SFT on the 137 fail prompts (distilled from a stronger solver) or RL with a correctness reward.
Script: not yet written. Will live under `scripts/train/` when the time comes.

### B3 — tool-augmented eval
Inject a calculator / Python executor at inference; measure how much of the gsm8k error is pure arithmetic vs. reasoning structure.
Script: not yet written.

### B4 — verifier / critic head
Add a lightweight head that scores the final answer before committing; single-pass self-correction.
Script: not yet written.

---

## Routing logic

```
A3 REJECTED  — conf-based revise dead at all token/span granularities
A4 SUPPORTED — block-layout rerank → 8.33% rescue
A5 SUPPORTED — prompt-template rerank → 13.33% rescue
A6 SUPPORTED — gen-length rerank → 20.00% rescue (strongest single axis)
A4x5 joint   — 6-cell run 16.67%, perfectly validates overlap prediction
H3 SUPPORTED — pass@N 23.33% rescue (capacity ceiling REJECTED)

Rescue set cross-analysis:
  A4∪A5 = 10, A4∪A5∪A6 = 13, full union = 15/18 = 83.3%
  Only 3 prompts (idx=5,6,16) unsalvageable by any method = true capacity ceiling
  Axes are mostly orthogonal: A6 contributes 3 unique, H3 contributes 3 unique
```

**Next-step priorities — per-prompt strategy search pipeline**:

The A axis is thoroughly swept; the full-method union of 83.3% shows the inference-time knob space is large enough. The next phase shifts from "which axis has signal" to "find the optimal configuration per prompt":

1. **Per-prompt strategy search**: for each prompt, search the optimal `(block_length × template × gen_length × temperature)` combination, store `(prompt, best_strategy)` pairs.
2. **New dimension: template_position**: a diffusion-LM-specific scaffold/inpainting approach — place template tokens at arbitrary positions within the generation region (not just as suffix). This is impossible with AR LMs.
3. **Train a strategy predictor**: use the strategy-search pairs to train a `prompt → best_strategy` model.
4. **P4/P6 offline analysis**: defer to N=137 (at N=60 all 7 features are non-significant).
