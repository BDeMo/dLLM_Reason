# Finding: A4 × A5 overlap — two mostly-independent levers, `answer` template is a trade, not a free win

> Language: English  |  中文: [finding_a4x5_overlap.zh.md](finding_a4x5_overlap.zh.md)

**Date**: 2026-04-15
**Companion**: [`finding_a_axis_exploration.md`](finding_a_axis_exploration.md)
**Script**: `scripts/validate/a4x5_overlap.py`
**Report**: `runs/validation/a4x5_overlap_182338_191434.json`

---

## TL;DR

On the 60-prompt common subset of A4 (`a4_block_rerank_20260415_182338`) and A5 (`a5_prompt_template_20260415_191434`):

| Metric | Value |
|---|---|
| `base_correct` (= A5 baseline = A4 bl32) | 42/60 |
| fail set size | 18/60 |
| A4 rescue | 5  (27.78% of fails) |
| A5 rescue | 8  (44.44% of fails) |
| **A4 ∩ A5** | **3** |
| A4 only | 2 |
| A5 only | 5 |
| **A4 ∪ A5** | **10  (55.56% of fails)** |
| independence factor | **0.769** (1.0 = fully disjoint; 0.5 = full overlap) |
| Joint 20-cell ensemble ceiling | 52/60 = **86.67% any-correct**, **55.56% of fails rescued** |

**Two main findings**:

1. **Levers are mostly independent.** 0.769 is close enough to 1.0 that a `layout × template` 20-cell ensemble is a real multiplicative win, not a redundancy. You rescue 10 distinct fails, not ~6 like full overlap would give.
2. **`answer` template is not a free upgrade.** Per-prompt inspection shows `answer` alone rescues 8 fails *and breaks 5 baseline-correct prompts* (net +3). The archive's earlier "`answer` prefix is immediately usable without ensembling" is **wrong**; it's a 3-for-8 trade that only nets positive because A5's rescues outnumber its breaks. The ensemble's `broken=0` hides this — baseline catches back the 5 that `answer` drops.

---

## Rescue-set detail

```
idx   gt       tag       winning_layouts                  winning_templates
 8    18       A5 only   -                                [answer]
10    125      A5 only   -                                [cot_step, answer]
13    15       A4 only   [bl64]                           -
15    8        BOTH      [bl64, short_then_long]          [cot_plain, cot_step, answer]
28    40       BOTH      [bl16]                           [cot_plain, answer]
35    48       A5 only   -                                [cot_plain, cot_step, answer]
48    623      A5 only   -                                [answer]
53    4        A4 only   [bl8]                            -
55    5        A5 only   -                                [answer]
59    3        BOTH      [bl8, bl64, short_then_long]     [cot_plain, cot_step, answer]
```

Three patterns worth flagging:

- **A5-only rescues skew to `answer`**: of the 5 A5-only rescues, 4 include `answer` and 3 are `answer`-exclusive. `answer` is A5's primary lever.
- **The 3 BOTH prompts are "easy rescues"**: they're the ones where almost any perturbation works — 2 of 3 have ≥ 3 winning templates and ≥ 2 winning layouts.
- **A4-only rescues are layout-specific**: idx=13 (bl64), idx=53 (bl8). Finest and coarsest uniform — there's no "bl32 is always close enough" pattern here.

---

## Per-config unique contribution

| Config | # fails rescued (when this config is correct AND baseline wrong) |
|---|---|
| A4.bl32 | 0 (definitionally; it IS the baseline) |
| A4.bl8 | 2 |
| A4.bl16 | 1 |
| A4.bl64 | 3 |
| A4.short_then_long | 2 |
| A5.baseline | 0 (definitionally) |
| A5.cot_plain | 4 |
| A5.cot_step | 4 |
| **A5.answer** | **8** |

`A5.answer` rescues all 8 of A5's fails. Removing it collapses A5 from 13.33% to ~6% rescue. It is *the* A5 lever.

---

## Correcting an earlier claim — `answer` as a single-template swap

Earlier finding doc said:

> **Can we ship `answer` template without ensembling?** `answer=45` vs `baseline=42` means the single-template switch is already net-positive. Cheapest possible gain.

The arithmetic is correct but the framing was wrong. Direct inspection of the per-prompt file:

```
baseline=T, answer=T:   37
baseline=T, answer=F:    5   ← broken by answer
baseline=F, answer=T:    8   ← rescued by answer
baseline=F, answer=F:   10
```

So swapping `baseline → answer` trades 5 correct-to-wrong for 8 wrong-to-correct. Net +3. On a 60-prompt fail-enriched slice this is positive, but:

- On the general population (which is heavily `base_correct=T`), breaking 5/42 passing prompts is **11.9% regression on baseline-passing prompts**.
- On the overall gsm8k eval (where baseline-correct ratio is ≈ 0.85+), this almost certainly *loses* accuracy globally.

**Correct recommendation**: use `answer` only as part of the ensemble (let `baseline` catch back the 5 broken prompts), not as a drop-in replacement. Or equivalently: use a 2-template ensemble of `{baseline, answer}` — cheapest configuration that captures essentially all of A5's rescue signal at 2× inference instead of 4×.

The broken-by-`answer` prompts (by idx, gt):
```
idx=2  gt=64
idx=17 gt=104
idx=22 gt=48
idx=24 gt=163
idx=57 gt=26
```

Inspection hint for a later session: are these prompts where the reasoning chain is non-trivial enough that cutting to `"Answer:"` suppresses the arithmetic the model needs to commit first? Numeric answers are ordinary — not outsized. No obvious structural tell from idx+gt alone.

---

## Joint 20-cell ensemble — upper bound, not a plan

Running all 5 layouts × 4 templates = 20 decodes per prompt would rescue 10/18 = 55.56% of fails on this sample, vs 44.44% (A5 alone) or 27.78% (A4 alone). This is a **real lift** — A4 adds 2 unique rescues on top of A5.

But 20× inference is not shippable. The realistic compression:

- **`{baseline, answer}` × `{bl8, bl32, bl64}` = 6 cells.** Keeps A5's dominant lever (`answer`) plus its safety net (`baseline`), plus A4's most-contributing layouts.
  - Expected rescue: at least 8 (from A5 ensemble) + idx 13 (bl64-only) + idx 53 (bl8-only) = **10 = same as full 20-cell**, assuming no new interactions appear.
  - ~~This is the configuration to try next if A4 × A5 pays rent.~~ **Done — see below.**

- **`{baseline, answer}` alone = 2 cells.** Expected rescue: 8 — captures all of A5's signal, drops A4's 2 independent rescues (idx 13, 53). Probably the cheapest deployable config.

### Joint 6-cell validation (post-closure addition)

`{baseline, answer} x {bl8, bl32, bl64}` 6-cell actual run:

```
N=60  base=42  any=52  rescued=10  rescue_rate=16.67%

per-cell:
  bl8_baseline  = 43
  bl8_answer    = 41
  bl32_baseline = 42  <- original baseline
  bl32_answer   = 45
  bl64_baseline = 37
  bl64_answer   = 40
```

**Predicted 10 rescues, got 10 rescues, 100% match, zero surprises.**

This validates three things:
1. **Overlap analysis methodology is reliable**: predicting joint results from independent per-prompt experiments hits perfectly.
2. **No unexpected cross-configuration interference**: different layout x template combos don't break each other.
3. **6-cell is a genuinely deployable configuration**: 6x inference captures the same rescue as the theoretical 20-cell ceiling.

---

## What this says about A4.1 (per-prompt layout predictor)

A4.1's motivation was: "if we can predict which layout per prompt, we avoid paying 5× at eval time."

After this overlap analysis:

- **A4's total independent rescue signal on this run is 2 prompts** (idx 13, 53 — A4-only). Of A4's 5 rescues, 3 are also captured by A5.
- A predictor that gets idx 13 → `bl64` and idx 53 → `bl8` perfectly would add 2/60 = 3.33% over an A5 ensemble alone. On N=137 scaled up, maybe 5-8 prompts.
- This is a **small** marginal signal to train a predictor on. 2 positive examples (and 58 negatives if we frame as binary "does layout ≠ bl32 matter") is not enough for a learned classifier; a hand-crafted feature (prompt length, number of numeric tokens, operator count, etc.) might catch it — or might just overfit.

**Verdict on A4.1 priority**: **lowered**. Spend the effort on:

1. Resume A4/A5 to N=137 first (locks in independence factor and A4-only rescue count).
2. Try the `{baseline, answer}` 2-cell ensemble on the full eval set — verify the global-regression prediction (answer-alone loses on non-fail prompts) is wrong or right.
3. Only then, if A4-only rescues still matter at scale, prototype a hand-crafted feature classifier for A4.1. Learned A4.1 from 2-8 positives is probably wishful thinking.

---

## Open questions

1. **Does the independence hold on N=137?** With only 18 fails on N=60, overlap statistics are noisy. The 0.769 factor could drift significantly on a larger sample. `--resume` both runs, then rerun this script.
2. **Does `answer` break *the same 5* prompts that other templates rescue?** If yes, the failure mode of `answer` is specific and potentially detectable. If the broken-by-`answer` prompts are scattered, there's no pattern to exploit.
3. **20-cell ensemble actual rescue vs upper bound**: if we actually ran all 20 configs, would rescue come in at the full 55.56%? The upper bound assumes best-case selection (oracle knows which config works); a realistic voting/majority scheme will likely undershoot.
4. **H3's 1/17 rescue**: is that prompt in the union_rescue list (idx ∈ {8, 10, 13, 15, 28, 35, 48, 53, 55, 59})? Cross-referencing with H3's per_prompt output (when H3 finishes) is one line of code.
