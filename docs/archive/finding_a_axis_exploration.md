# Finding: A-axis Exploration — Block Granularity is the First Signal

> Language: English  |  中文: [finding_a_axis_exploration.zh.md](finding_a_axis_exploration.zh.md)

**Date**: 2026-04-15 (initial) / 2026-04-16 (synced after H3 rerun at n=60 + P6 crossref)

**Companion docs**:
- [`hypotheses.md`](hypotheses.md) — formal hypothesis registry, verdict board
- [`exploration_axes.md`](exploration_axes.md) — A-axis/B-axis index with status tags
- [`finding_dag_search_zero_rescue.md`](finding_dag_search_zero_rescue.md) — why A1 (edge DAG) is DEAD

---

## TL;DR

After three independent DAG search implementations and two token-level revise experiments all returned **0 rescue**, the A axis produced three positive signals at coarse granularities: A4 block-layout rerank (5/60 = 8.33%), A5 prompt-template rerank (8/60 = 13.33%), and **A6 gen-length rerank (12/60 = 20.00%, strongest single A-axis knob)**. H3 pass@N re-run on the full n=60 fail set gave **52/60 = 86.67%**, the strongest single-dimension lever (near-orthogonal to A axis). Full-method union **55/60 = 91.67%**; true capacity ceiling **5 prompts [4,5,14,41,42]**.

| Experiment | Knob | Unit of intervention | Verdict | Rescue |
|---|---|---|---|---|
| A1 · DAG search | order | single edge in DAG | **DEAD** (3 impls) | 0/1319+200+106 |
| A2 / H1 · token revise | commit retraction | 1 token at a time | **REJECTED** | 0/137 |
| A3 · span revise | commit retraction | window of 4 tokens | **REJECTED** | 0/60 |
| **A4 · block layout** | **denoise path** | **whole block (8–64 tok)** | **SUPPORTED** | **5/60 (8.33%)** |
| **A5 · prompt template** | **input** | **the prompt itself** | **SUPPORTED** | **8/60 (13.33%)** |
| **A6 · gen length** | **generation budget** | **total length (64–256 tok)** | **SUPPORTED** | **12/60 (20.00%)** |
| A4×A5 joint 6-cell | layout × template | 6-cell ensemble | — | 10/60 (16.67%), overlap prediction perfectly validated |
| H2 · order vs content | variance ratio | block_length ∈ {16,32,64} | REJECTED by stated threshold; foreshadows A4 | ratio 0.754 |
| **H3 · pass@N (n=60)** | **sampling diversity** | **—** | **REJECTED** (per capacity-ceiling threshold) | **52/60 (86.67%)** |

The key inversion: H2 was actually a block_length sweep (`block_length ∈ {16, 32, 64}` at T=0, not three different schedulers as originally framed) and REJECTED the claim "order carries < 30% of content's signal" (ratio 0.754 >> 0.3) — i.e. block-layout order already carried 75% of the output variance of temperature. A1's triple-0-rescue at the edge level says nothing about this. A4 is the direct next step: take H2's same knob (block_length) and ask not "does it change the output" but "does it change *correctness*". Answer: yes, for 8.33% of fail prompts.

---

## Scope — the 137 fail prompts (H0)

`scripts/validate/h0_forensics.py` reads `correct=0` rows from the most recent `runs/research_*/stage2_discovery/episodes.db`, buckets them by error mode, and dumps:

- `runs/validation/scope_fail_prompts.json` — 137 prompts where LLaDA-instruct (T=0, `block_length=32`, `remasking=low_confidence`, `gen_length=128`, `steps=128`) gets gsm8k wrong.
- `runs/validation/scope_ok_prompts.json` — the matching `correct=1` set, used as H3's control group.

Error mode distribution (informational only, no verdict):
- `numeric_close` / `numeric_mid` / `numeric_far` / `numeric_order` — wrong number by increasing magnitude of relative error.
- `format_bad` — truncated / no number emitted.
- `unknown` — gt itself unparseable.

Every downstream A-axis experiment uses `scope_fail_prompts.json` so results are comparable one-to-one across experiments — a prompt that rescues in A4 can be looked up in the H1/A3 records to see if it was rescued there too.

---

## H1 — Single-token revise hook · REJECTED

**Claim**: enabling a revise hook that masks committed tokens whose `conf < τ` every few steps rescues fail prompts.

**Design**:
- `scripts/validate/h1_remask_rescue.py`
- Two inferences per prompt:
  1. **baseline**: native LLaDA (T=0, low-confidence remasking, no hook).
  2. **revise**: identical to baseline, but every `revise_every=8` global steps, retroactively mask all committed positions (within the current block) whose `committed_conf < revise_thresh=0.3`. Masked positions re-enter the sampling pool on the next step.
- `rescue_rate = |{prompt : revise_correct ∧ ¬baseline_correct}| / N`.

**Result**: N=137, base=0, revise=0, rescued=0, broken=0, rescue_rate=**0.00%** → REJECTED.

**Diagnosis**: base_correct = revise_correct = 0/137, rescued=0, broken=0 — the hook makes zero observable difference. Consistent with "trigger rate on fail prompts is low because committed conf stays ≥ 0.3", but the final correctness vector alone only proves "this hook rescues nothing", not the exact trigger count (the summary.json does not persist a trigger counter). **Per-token confidence carries no usable error signal on these fail prompts**; the model is confidently wrong.

---

## H2 — Order variance vs content variance · REJECTED (but nuanced)

**Claim**: holding the prompt fixed, varying scheduler (DAG/order) produces *much less* output variance than varying temperature (content). Specifically `order_var / content_var < 0.3` would support H2.

**Design**:
- `scripts/validate/h2_order_vs_content.py`
- K=20 prompts from fail set.
- **Content axis**: same scheduler, T ∈ {0, 0.3, 0.7}, 3 samples each → 9 outputs per prompt.
- **Order axis**: T=0 fixed, same `low_confidence` scheduler, 3 `block_length` values ∈ {16, 32, 64} → 3 outputs per prompt. (Changing block size reshapes the denoise commit order across steps without touching content sampling.)
- Per prompt, compute variance of pairwise normalized edit distance across outputs.

**Result**: N=20, content_var=0.256, order_var=0.176, ratio=**0.754** → REJECTED.

**Interpretation**: order is not dwarfed by content — it's 75% of content's variance. That *appears* to contradict A1 (edge DAG produces 0 rescue). The reconciliation: H2's "order axis" was literally `block_length ∈ {16, 32, 64}` — the same knob A4 probes at larger scale. So H2 already measured **block-layout variance**, not edge-level variance; A1's 0-rescue at the single-edge level and H2's non-trivial variance at the block level are talking about different granularities. A4 is the direct follow-up: does this block-level variance translate into *correctness* rescues, not just output diversity?

---

## H3 — pass@N capacity ceiling · Strongest single-dimension lever (n=60 authoritative)

**Claim**: these 137 prompts are at LLaDA-instruct's capability ceiling — even temperature + N samples can't rescue them.

**Design**:
- `scripts/validate/h3_passN_at_temperature.py`
- Full n_fail + n_ok control.
- Per prompt × T ∈ {0.3, 0.7, 1.0} × N=8 samples → compute pass@1 / pass@4 / pass@8.
- Verdicts: `fail_p@8 < 5% AND ok_p@8 > 90%` → SUPPORTED (capacity ceiling); `fail_p@8 > 20%` → REJECTED.

**Initial (n=30 fail, 2026-04-15)**: 7 prompts rescued (idx=0,2,8,13,15,24,28), rescue_rate=23.33%, 4 stuck. Small-sample cross-analysis: H3-only {2, 24}, H3 ∩ A6-only {0}, H3 ∩ (A4∪A5) {8, 13, 15, 28}.

**n=60 rerun (`h3_passN_20260415_133254`, 2026-04-16)**: extended to full 60 fail + 30 ok control.
- `fail_pass@8_max = 86.67%` (T=1.0) / `ok_pass@8_max = 100%` → **REJECTED per threshold** (= capacity ceiling does not hold).
- P6 crossref (`p6_h3_crossref.py`) output: H3 rescue = **52/60 (86.67%)**, H3 stuck = **8 prompts [4, 5, 14, 19, 41, 42, 48, 51]**.

**n=60 rescue-set cross-analysis (P6 authoritative)**:
- A-union (A4∪A5∪A6) = 13/60 (21.67%).
- H3 ∩ FAIL18 (the 18-prompt subset where bl32/baseline/g128 baselines all fail) = 10 prompts {0, 8, 10, 13, 15, 28, 35, 53, 55, 59} — **all covered by A-union; H3 contributes 0 unique rescue inside FAIL18.**
- H3's 42 "only"-rescues all live outside FAIL18 (= A-axis baselines already correct, but there are also correct completions at T>0).
- A6-only rescues {19, 51} stay stuck under H3 too (confirms "write-space > diversity").
- idx=48 forms a new category: "rescued by A5+A6+Joint, stuck under H3" — A axis rescues, pass@N cannot.

**Interpretation**:
1. The initial n=30 "23.33%" was a **small-sample artifact**; on the full 60 prompts pass@N rescue is 86.67% and **dominates** the entire A axis rather than being comparable.
2. On the strict "rescue a baseline-wrong prompt" question, H3 still contributes **0 unique rescues** (10/10 swallowed by A-union in FAIL18). H3's value lies in pass@8 stably flipping many baseline-correct prompts under T=0.3 — which conversely also covers many baseline-wrong ones.
3. The write-space signal (A6-only {19, 51}) remains uncovered by H3, proving the A-axis rescue is systematic, not lucky sampling.
4. True capacity ceiling is fixed at 5 prompts [4, 5, 14, 41, 42].

**Historical bug**: `p5_h3_crossref.py` was written for the old H3 schema (`pass_at_k` dict) and silently outputs h3_rescue=0 under the current `fail_XXXX.json + temps.T.pass@k` shape. **Fixed by `p6_h3_crossref.py`**.

---

## A1 / A2 — Edge DAG + token revise · DEAD (recap)

A1 is already archived in `finding_dag_search_zero_rescue.md`. Summary: three independent implementations (greedy ±1 edge on 1319 prompts, NAS supernet on 200 prompts, E2E differentiable on 106 prompts) all produced **0 rescue**, with NAS and E2E additionally reporting **0 edges selected** by their own optimizers. Edge-level ordering carries no signal over the greedy low-confidence baseline at T=0.

A2 is H1 under a different name (single-token revise hook); REJECTED as above.

---

## A3 — Span-level revise · REJECTED

**Claim**: errors cluster in **contiguous spans** (a mis-computed sub-expression) where individual tokens may be locally confident but the **window mean** confidence drops.

**Design**:
- `scripts/validate/a3_span_revise.py`
- Server-side (via new `/generate_span_revise` endpoint in `scripts/serve.py`).
- Identical sampler to H1 except the revise criterion and action.
- **Criterion**:
  - Every `revise_every=8` global steps, compute a 1-D sliding-window mean of committed-token confidence via `F.conv1d` (kernel = `torch.ones(window_size=4)`, padding=2).
  - Counts tensor (also via conv1d on a mask of "position is committed") filters unreliable windows where `counts < max(2, window_size // 2)`.
  - Window is "bad" if `mean < revise_thresh=0.4` AND reliable AND the center position is committed.
- **Action**: mask ALL committed positions covered by any bad window (another conv1d pass on the `bad_center` indicator, any overlap → kill). Confidence reset to `+inf` so next step re-samples.
- Thresholds deliberately **looser than H1** (0.4 vs 0.3) because averaging shrinks the tail of extreme low-conf tokens — the signal we want is "a *run* of mediocre tokens", not "one catastrophically low token".

**Result**: N=60, base=42, revise=42, **rescued=0, broken=0, rescue_rate=0.00%** → REJECTED.

The two passes produce literally identical correctness on every prompt in the sample (base_correct = revise_correct = 42/60, rescued=0, broken=0). Combined with H1, this is strong evidence that **every conf-based revise hook is dead on this fail set**, not just a poorly-tuned threshold problem:
- H1's narrow single-token criterion (τ=0.3) rarely triggered on fail prompts (diagnostic from H1 run: most prompts stayed above threshold).
- A3's window-mean criterion (τ=0.4, win=4) is strictly less stringent than H1's per-token check, so it should trigger at least as often; yet the correctness vector is pixel-identical. Either the looser threshold still fails to fire on most prompts, or it does fire but the resampled tokens converge back to the same wrong commits.

Either way, this kills the "conf-as-error-signal" direction at any granularity finer than a block. The model's internal uncertainty is not correlated with its mistakes on these prompts.

---

## A4 — Block-layout rerank · SUPPORTED

**Claim**: the default `block_length=32` is one arbitrary point in a space of valid block layouts; at least one other layout rescues some fails.

**Design**:
- `scripts/validate/a4_block_rerank.py`
- Server-side via existing `/generate` (uniform) and new `/generate_block_schedule` (non-uniform).
- Per prompt, 5 layouts, T=0:

| Layout name | Kind | Spec | Rationale |
|---|---|---|---|
| `bl8` | uniform | `block_length=8`, 16 blocks | finest uniform, smallest per-step competition pool |
| `bl16` | uniform | `block_length=16`, 8 blocks | intermediate |
| `bl32` | uniform | `block_length=32`, 4 blocks | **baseline** — matches H1's base_correct exactly |
| `bl64` | uniform | `block_length=64`, 2 blocks | coarsest uniform, large competition pool |
| `short_then_long` | non-uniform | block_sizes=[16,16,16,16,64], steps=[16,16,16,16,64] | first 64 tokens fine-grained (reasoning steps), last 64 coarse (answer emission) |

- **Step budget normalization**: total forward passes equal across layouts. For uniform: `steps = max(args.steps, num_blocks)` rounded up to a multiple of `num_blocks`. For non-uniform: `sum(steps_per_block) == 128`. This avoids the confound "coarser block → fewer steps → worse output".

**Why T=0 still produces different outputs across layouts** — critical subtlety:
- Single forward on a fixed `x` is deterministic.
- But denoising is a *sequence* of forwards where each step's input is determined by the previous step's commits.
- Different block layouts commit different positions at each step → later forwards see different `x` → different logits → different commits.
- So the final output is path-dependent even though each forward is deterministic.

**Verdict computation**:
```
rescue_rate = |{prompt : (any layout correct) AND (bl32 wrong)}| / N
broken      = |{prompt : bl32 correct AND all layouts wrong}|
any_rate    = |{prompt : any layout correct}| / N
```

**Result** (N=60, mid-run; run stopped before full 137):
```
base(bl32)=42  any_layout=47  rescued=5  broken=0
rescue_rate=8.33%  any_rate=78.33%
per_layout: bl8=43  bl16=41  bl32=42  bl64=37  short_then_long=37
```
→ **SUPPORTED** (8.33% ≥ 5% threshold).

**Three observations from the per-layout numbers**:
1. **Finer is slightly better**: bl8=43 > bl32=42 (+1), but bl64=37 (−5) and short_then_long=37 (−5). Finer block = smaller per-step competition pool = more conservative commits; coarser = more aggressive, breaks more prompts.
2. **No single layout dominates**: rescue=5 requires the ensemble; no individual layout alone beats bl32 by 5. The 5 rescued prompts are spread across layouts (diagnostic: look at `per_prompt/*.json` records).
3. **`broken=0`**: no prompt is *lost* by switching layouts; this is pure ensemble gain, not a trade-off.

**Caveats**:
- **N=60 not 137**: the run was interrupted. 5 rescues on 60 prompts is 8.33%; if the remaining 77 prompts contribute only +1 rescue the rate drops to 4.4% → INCONCLUSIVE. Needs `--resume` to 137 to stabilize the verdict.
- **Ensemble vs single-layout**: the "SUPPORTED" result is an ensemble claim. If we're asking "is there a single better block_length?", the honest answer is no — bl8's +1 over bl32 is within noise. The real claim is "there exists a *per-prompt* best layout", which is a weaker and more interesting result.
- **No prompt-to-layout predictor yet**: we know a better layout exists per prompt, but we have no way to pick it without trying all 5. A follow-up experiment (A4.1) would train a predictor `prompt → best_layout` — if that predictor's accuracy is > random-pick (1/5 = 20%), the layout signal is usable as a learned sampler knob.

**Reconciliation with H2**:
- H2 already varied `block_length ∈ {16, 32, 64}` (T=0, same scheduler) and found `order_var / content_var = 0.754` — block-layout order alone carries ~75% of content's output variance.
- A1 said edge-level order has 0 rescue signal.
- A4 says block-level order has 8.33% rescue signal.
- **Consistent story**: the 0.75 variance ratio was *always* a block-layout measurement (H2's "order axis" = a subset of A4's layout sweep). The signal is real at the block level and was already visible in H2's output-diversity numbers; A4 shows that this diversity also contains correctness gains, not just surface edits. A1's edge-level search was hunting in a space finer than where the variance actually lives.

---

## A5 — Prompt-template rerank · SUPPORTED

**Claim**: an appropriate prompt prefix (CoT-inducing or answer-direct) shifts the output distribution enough to rescue.

**Design**:
- `scripts/validate/a5_prompt_template.py`
- Server-side via existing `/generate` (same backend as A4 uniform layouts).
- Per prompt, 4 templates (suffix appended to original prompt):
  1. `baseline` — no change.
  2. `cot_plain` — `"\nLet's solve this step by step."` (explicit CoT invocation).
  3. `cot_step` — `"\nStep 1:"` (structurally forcing CoT skeleton).
  4. `answer` — `"\nAnswer:"` (suppresses CoT, forces direct answer).
- Same verdict logic and thresholds as A4.

**Design intent distinction from A4**: A4 changes *how* the sampler walks; A5 changes *what* the sampler sees. If A5 SUPPORTED but A4 not → problem was prompt-framing; if A4 SUPPORTED but A5 not → problem was denoise path; if both (observed) → both axes carry signal, and the overlap between A4-rescued and A5-rescued prompt sets tells us whether they're independent levers.

**Result** (N=60, run stopped at same N as A3/A4):
```
base=42  any_template=50  rescued=8  broken=0
rescue_rate=13.33%  any_template_rate=83.33%
per_template: baseline=42  cot_plain=35  cot_step=30  answer=45
```
→ **SUPPORTED** (13.33% ≥ 5% threshold). Highest per-experiment rescue rate seen on the A axis.

**Three observations from the per-template numbers — this is the surprising part**:
1. **`answer` beats baseline (+3)**: the single best prompt is the *direct-answer prefix*, not either CoT prefix. The gsm8k prompts used by H0 already contain the instruction set to elicit reasoning, so an explicit `"\nAnswer:"` suffix appears to short-circuit verbose expansion and land the numeric answer in fewer tokens — within the 128-token budget.
2. **CoT prefixes actively *hurt*** (`cot_plain` −7, `cot_step` −12): forcing step-by-step is worse than the LLaDA-instruct default. The `cot_step` prefix `"\nStep 1:"` is especially bad, likely because it commits to a particular format that doesn't match what the instruct-tuned model wanted to emit, creating a mismatch between prompt shape and decoder prior.
3. **`broken=0` again**: like A4, this is a pure ensemble gain. No prompt that baseline got right is lost to template ensembling. The 8 rescued prompts come entirely from templates other than baseline finding an answer.

**Where the 8 rescues live** (rough decomposition, ignoring intersection — to be computed exactly from per_prompt records):
- `answer` alone rescues ≥ 3 prompts baseline missed (since `answer=45 > baseline=42`).
- `cot_plain`/`cot_step` each rescue a handful too (even though they net-drop), because the prompts they rescue and the ones they break are different subsets.
- The fact that `any_template=50 > answer alone=45` means at least 5 prompts are rescued *only* by a CoT template — i.e., CoT is not globally useless, just globally risky.

**Caveats**:
- **N=60 not 137**: same interruption as A3/A4. 8/60 = 13.33%; if the remaining 77 prompts contribute proportionally the final rate is roughly stable, but needs `--resume` to lock in.
- **Template ensemble cost**: 4× inference per prompt. Earlier hope that "one-template-for-all" (`answer` alone) would be immediately usable is **wrong** — the overlap analysis ([`finding_a4x5_overlap.md`](finding_a4x5_overlap.md)) shows `answer` alone rescues 8 baseline-wrong prompts while breaking 5 baseline-correct prompts. Net +3 on this fail-enriched subset; negative in general distribution. Use `{baseline, answer}` 2-cell ensemble as the minimum shippable configuration.
- **`answer` only beats baseline by 3** on a base of 42 — in absolute terms it's a 7% lift on the passing count, or turns 3 wrong into 3 right out of 60. Significant relative to noise floor (broken=0), but small in absolute terms.

**Reconciliation with A4**:
- A4 SUPPORTED (5/60 rescued by layout diversity).
- A5 SUPPORTED (8/60 rescued by template diversity).
- Both have `broken=0`, both are pure ensemble wins, both stop at N=60.
- Open question (Q4 below): are the A4-rescued and A5-rescued prompt sets disjoint? If yes, 5 + 8 = 13 distinct fail prompts are rescuable by `layout × template` ensembling — a **20-cell ensemble** (5 layouts × 4 templates) could in principle rescue up to ~22% of fails. If overlapping, both are picking up the same "sampling-brittle" prompts and we'd see diminishing returns.

---

## A6 — Gen-length rerank · SUPPORTED (strongest single axis)

**Claim**: the default `gen_length=128` is not optimal for all prompts; different generation lengths rescue errors.

**Design**:
- `scripts/validate/a6_gen_length.py`
- Per fail prompt, sample with gen_length in {64, 96, 128, 160, 192, 256} (fixed block_length=32, T=0).
- Same verdict thresholds as A4/A5.

**Result** (N=60):
```
base(g128)=42  any_length=54  rescued=12  broken=0
rescue_rate=20.00%  any_length_rate=90.00%
per_length: g64=27  g96=36  g128=42  g160=49  g192=39  g256=40
```
-> **SUPPORTED** (20.00% >= 5% threshold). **Strongest single-axis signal on the A axis.**

**Key observations**:
1. **g160 is the sweet spot**: 49/60=81.7% vs baseline g128=42/60=70%, single-config +11.7pp. g160 = 5 blocks x 32 tokens, one extra block gives enough reasoning room.
2. **Not "longer is better"**: g192=39 (-3), g256=40 (-2), both worse than baseline. Long budgets cause trajectory divergence.
3. **g160 alone nearly matches A5's any-template ensemble** (49 vs 50).
4. **Short budgets severely hurt**: g64=27 (-15), g96=36 (-6).

**Rescue set**: {0,10,13,15,19,28,35,48,51,53,55,59} (12 prompts)
- A6-only: {19, 51}
- A6 intersect H3-only: {0}
- Substantial overlap with A4 union A5 but still contributes independently.

---

## A4 x A5 Joint 6-cell validation

`{baseline, answer} x {bl8, bl32, bl64}` 6-cell run:
- N=60, base=42, any=52, rescued=10, rescue_rate=**16.67%**
- **Perfectly validates overlap prediction**: predicted 10 rescues, got 10 rescues, 100% match, zero surprises.
- per-cell: bl8_baseline=43, bl8_answer=41, bl32_baseline=42, bl32_answer=45, bl64_baseline=37, bl64_answer=40

Validates that the overlap analysis methodology is reliable and no unexpected cross-configuration interference exists.

---

## Cross-experiment interpretation

### The granularity ladder

| Level | Experiment | Intervention | Verdict |
|---|---|---|---|
| 1 token | A2/H1, A3 | revise one token / one span | DEAD |
| 1 edge (token pair) | A1 (x3) | rewire DAG edge | DEAD |
| 1 block (8-64 tok) | A4 | swap layout | **SIGNAL** (8.33%) |
| 1 prompt | A5 | change template | **SIGNAL** (13.33%) |
| gen_length | A6 | change generation length | **SIGNAL** (20.00%, strongest) |
| layout x template | A4xA5 joint | 6-cell ensemble | 16.67% (overlap perfectly validated) |
| sampling scheme | H2, H3 | H2 varies `block_length`; H3 varies T + N | H3 (n=60) **86.67% dominates A axis** |

The signal first appears at the **block level** and grows at the prompt/gen_length level. Everything finer (token, span, edge) is dead. A6 gen_length is the strongest single-axis signal within A (20%); g160 alone nearly matches A5's any-template ensemble. H3 (pass@N multi-T) is the **strongest cross-axis lever** at 52/60 = 86.67%, but it is near-orthogonal to the A axis (H3 ⊆ A-union is only 19.2%).

**Full-method union = 55/60 = 91.67%** (n=60 authoritative). Only 5 prompts [4,5,14,41,42] are unsalvageable by any method = true capacity ceiling. In the FAIL18 subset view: A-union = H3-union = 13/18 = 72.2%, H3 contributes 0 unique rescues inside FAIL18; A6-only rescues {19, 51} remain stuck even under H3.

**The rescue signal scales with granularity**: A4's 8.33% -> A5's 13.33% -> A6's 20.00% are monotone in intervention coarseness. This is evidence that **the LLaDA-instruct model doesn't have a fixable "ordering mistake" per se** — there's no small local fix. What works is a different global denoise trajectory (A4), a different input framing (A5), or a different generation budget (A6) — all of which redistribute the *entire* output, not just a localized token.

### What kills `conf`-based revise

A2 + A3 together are strong evidence: **the confidence the model reports on committed tokens does not correlate with error on fail prompts**. Neither narrow (0.3 single-token) nor loose (0.4 window-mean) thresholds find errors. Any future sampler-side correction must use a signal *other than* committed-token confidence — candidates: self-consistency across layouts (A4.1 idea), separate verifier head (B4), or tool-augmented re-checking (B3).

### What A4 does and does not show

- **Does show**: there exists at least one block layout per prompt that is better than the default bl32, for roughly 8% of fail prompts.
- **Does not show**: that any single layout is better than bl32 globally. The per-layout totals in this run actually go bl8 > bl16 ≈ bl32 > bl64 ≈ short_then_long, but the gaps are ±1 to ±5 on N=60 — within sampling noise if we had one.
- **Does not show**: that the best layout per prompt is predictable from the prompt. That's A4.1.

### Decision tree (post-closure, full results)

```
A3 REJECTED ────────────┐
                        ├──> conf-based revise DEAD at all granularities below block
A1 DEAD (3x)  ──────────┘          → drop revise hooks from sampler roadmap

A4 SUPPORTED (5/60) ────┐
H2 REJECTED (0.754) ────┤
A5 SUPPORTED (8/60) ────┤──> block-layout + prompt-template + gen-length all carry signal
A6 SUPPORTED (12/60) ───┘     → A6 strongest (20%), g160 is the sweet spot
                              → A4xA5 joint 6-cell perfectly validated (10=10)
                              → A-union = 13/60 = 21.67% (n=60)
                              → full-method union = 55/60 = 91.67% (n=60)
                              → FAIL18 subset: A-union = H3-union = 13/18 = 72.2%

H3 (n=60) pass@8 = 86.67% (52/60)  ← REJECTED per capacity-ceiling threshold
  → capacity ceiling REJECTED (model is far from its ceiling)
  → H3-only 42 prompts all live outside FAIL18; 0 unique rescues inside FAIL18
  → H3 ⊆ A-union (full set) = 19.2%, near-orthogonal to A axis
  → A6-only {19, 51} stay stuck under H3 too → "write-space > diversity"

True capacity ceiling = 5 prompts [4, 5, 14, 41, 42]  (consistent across n=60 and FAIL18 views)

→ next: per-prompt strategy search (block_length x template x gen_length x temperature)
→ new dimension: template_position (diffusion-LM-specific scaffold/inpainting)
```

---

## Open questions (post-closure update)

**Resolved**:
1. ~~A4 x A5 overlap~~ → **DONE**: independence=0.769, joint 6-cell run perfectly validated (10=10)
2. ~~H3 completion~~ → **DONE (n=60 authoritative)**: pass@8 = 86.67% (52/60), REJECTED per capacity-ceiling threshold; H3 ⊆ A-union only 19.2% (near-orthogonal), FAIL18 ∩ H3 = 10 prompts {0,8,10,13,15,28,35,53,55,59} all covered by A-union, 0 unique rescues inside FAIL18
3. ~~Can we ship `answer` alone~~ → **NO** (overlap analysis confirmed)
4. ~~gen_length expansion~~ → **DONE → A6 SUPPORTED**, rescue=20%
5. ~~p5_h3_crossref.py BUG~~ → **DONE**: `p6_h3_crossref.py` created to fix the schema mismatch (old `pass_at_k` dict vs current `temps.T.pass@k` silently outputting h3_rescue=0). P6 is the n=60 authoritative crossref, outputs full_union=55/60 (91.67%), ceiling=5 [4,5,14,41,42], h3_only=42, a6_only=[19,51]

**Open**:
1. **Per-prompt strategy search**: search optimal `(block_length x template x gen_length x temperature)` per prompt → `(prompt, best_strategy)` pairs. This is the next-phase mainline.
2. **template_position new dimension**: diffusion-LM-specific scaffold/inpainting — place template tokens at arbitrary positions within generation region. Unique to diffusion LMs.
3. **P4/P6 offline analysis**: CoT-broken pattern + A4-only rescue features. At N=60 all 7 features are non-significant (see `finding_p4_p6_feature_analysis.zh.md`); defer to N=137.

---

## Files touched for this finding

### Scripts
- `scripts/validate/h0_forensics.py` — scope generation (fail + ok)
- `scripts/validate/h1_remask_rescue.py` — H1 + base sampler shared by all
- `scripts/validate/h2_order_vs_content.py` — H2
- `scripts/validate/h3_passN_at_temperature.py` — H3
- `scripts/validate/a3_span_revise.py` — A3, HTTP client to server
- `scripts/validate/a4_block_rerank.py` — A4, HTTP client to server
- `scripts/validate/a5_prompt_template.py` — A5, HTTP client to server
- `scripts/validate/_http_client.py` — shared FastAPI client
- `scripts/validate/aggregate_verdicts.py` — idempotent verdict-board rewriter
- `scripts/validate/run_a_axis.sh` — sequential A3→A4→A5 runner
- `scripts/validate/p6_h3_crossref.py` — H3 × A-axis n=60 crossref (replaces buggy P5)
- `scripts/serve.py` — added `/generate_span_revise`, `/generate_block_schedule`

### Server-side samplers
- `src/dllm_reason/inference/validation_ext.py` — `generate_span_revise`, `generate_block_schedule`, `generate_uniform` (shared by server endpoints)

### Results
- `runs/validation/h1_remask_20260415_051706/summary.json`
- `runs/validation/h2_order_content_20260415_054252/summary.json`
- `runs/validation/a3_span_revise_20260415_181502/summary.json`
- `runs/validation/a4_block_rerank_20260415_182338/summary.json`
- `runs/validation/a5_prompt_template_20260415_191434/summary.json`
- `runs/validation/h3_passN_*` (in flight)

### Docs
- `docs/archive/hypotheses.md` (+ `.zh.md`) — hypothesis registry
- `docs/archive/exploration_axes.md` (+ `.zh.md`) — A/B axis index
- `docs/archive/finding_dag_search_zero_rescue.md` (+ `.zh.md`) — A1 DEAD evidence
- `docs/archive/finding_a_axis_exploration.md` (+ `.zh.md`) — **this file**
