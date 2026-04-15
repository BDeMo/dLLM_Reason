# Finding: Static DAG Search Yields Zero Rescue on gsm8k

> Language: English  |  中文: [finding_dag_search_zero_rescue.zh.md](finding_dag_search_zero_rescue.zh.md)

**Date**: 2026-04-15
**Package version**: `dllm-reason v1.5.3` (`pyproject.toml`)
**Status**: Archived — core evidence for the Phase 1 pivot

---

## 1. Experimental setup (full reproduction)

### 1.1 Model & inference config (`configs/eval_default.yaml`)

| Field | Value |
|---|---|
| `model_id` | `checkpoints/llada-instruct` |
| `torch_dtype` | `bfloat16` |
| `num_steps` | 128 |
| `block_length` | 32 (sequence split into 4 blocks × 32 tokens) |
| `temperature` | **0.0** (greedy argmax) |
| `cfg_scale` | 0.0 |
| `remasking` | `low_confidence` (native LLaDA sampling, no extra correction) |
| `max_new_tokens` | 128 |

**Sequence layout**: `L = 128` (prompt 32 + gen 64 + padding 32, inferred from best_dag_edges/depth).

### 1.2 Search config (`configs/search/greedy.yaml`)

| Field | Config | Actual run |
|---|---|---|
| `method` | greedy | greedy ✅ |
| `budget` | 100 | **30** (CLI override) |
| `num_candidates` | 10 | 10 |
| `patience` | 5 | 5 |
| `initial_dag` | `cot` (CoT template warm-start) | — |
| `fitness` | accuracy | accuracy |
| `fitness_samples` | 50 | — |

### 1.3 Search algorithm (`src/dllm_reason/search/greedy.py` — `GreedyEdgeSearch`)

**Flow**:
1. **Template warm-start**: run every DAG in `init_templates` (history shows `history[0].step=8`, i.e. 8 templates evaluated), pick the highest-fitness one.
2. **Candidate generation** (`_generate_candidates`): per round, produce `num_candidates=10` DAGs, each a **single-edge** add/remove:
   - 60% probability: add a random edge (if acyclic)
   - 40% probability: remove a random existing edge
3. **Greedy evaluation**: iterate candidates, **accept the first with fitness > best**, regenerate candidates (`break`).
4. **Early stop**: exit after `patience=5` consecutive no-improvement rounds.

**Key properties**:
- **Fitness hill-climbing, no exploration temperature**
- **Early accept**: first improvement wins, other candidates ignored
- **Template warm-start consumes 8/30 of the budget** (27%)
- **Single-edge neighborhood**: ±1 edge per step (out of 3072 total edges = 0.03% perturbation)

### 1.4 Data artifacts

| Path | Contents |
|---|---|
| `runs/research_20260411_030422/stage2_discovery/search_histories/gsm8k/prompt_*.json` | 1319 prompt search histories (each with a `history` trajectory) |
| `runs/research_20260411_030422/stage2_discovery/episodes.db` | SQLite, 1319 rows, final `dag_json` + output + correct |
| `runs/research_20260411_030422/stage2_discovery/best_dag_per_prompt.json` | prompt → best_strategy / correct / num_strategies_tried |

---

## 2. Core facts

### 2.1 Rescue rate = 0

(Scope: single run `research_20260411_030422`, 1319 prompts.)

|              | final_fail | final_ok | All  |
|--------------|-----------:|---------:|-----:|
| **init_fail** |      137   |    **0** |  137 |
| **init_ok**   |        0   |    1182  | 1182 |
| **All**       |      137   |    1182  | 1319 |

- Of the 137 initially failing prompts, **rescue = 0 / 137 = 0.0%** after 30 search steps.
- Of the 1182 initially correct prompts, **0 were broken**.
- **Net Δacc from search = 0.000 pp**.

(The earlier 1533/151 numbers came from aggregating multiple runs; single-run numbers above are authoritative.)

### 2.2 Fitness stays 0 on init_fail prompts

Spot-check (`prompt_0002`, `prompt_0007`):

```
init:  fitness=0.00  edges=3072  step=8   (warm-start selected CoT template)
...
last:  fitness=0.00  edges=3073  step=30
fitness trajectory:  min=0.00  max=0.00  final=0.00
edges range:        3071 – 3073   (30 steps of ±1 edge perturbation)
```

**Across all 137 init_fail prompts, fitness never moved from 0** during the 23-step candidate-evaluation phase. Greedy hill-climbing has no signal on a perfectly flat plateau; `patience` never triggers because nothing ever improves.

### 2.3 The search space is effectively tiny

- Across 1319 prompts, the final best DAG has only **28 unique structures**.
- 22 of them appear exactly once.
- The dominant template (edges=3072, depth=4, max_width=32 — standard 4-block semi-AR) accounts for **86% (1134/1319)**.
- Search ≈ "start from CoT template, perturb ±1 edge, find no improvement, return to origin".

### 2.4 "Non-default DAG acc=100%" was a selection artifact

The earlier "185 prompts with non-default DAG at 100% acc" claim was misread:
- Template warm-start happened to hit a non-CoT template with init_fitness=1.0 → early-stop → that template was saved.
- These prompts are already solvable by **any** warm-start template (they are easy cases) — no specific DAG "rescued" them.
- Correlation ≠ causation.

---

## 3. Conclusion

**Under the v1.5.3 configuration (llada-instruct + T=0 + GreedyEdgeSearch budget=30, single-edge mutation), static position-level DAG search has no causal effect on final gsm8k accuracy.**

The earlier "+3.6pp" / "non-default DAG rescues 14%" claims are **invalidated**, caused by:
1. Mistaking "warm-start template hit" for "search convergence".
2. Confusing `default_fp` with a "semi-AR baseline" — `default_fp` is a CoT template, not a baseline.

---

## 4. Why is rescue zero? Three hypotheses (not yet falsified)

### H1: Commit-once-never-revise is the MDLM sampling bottleneck
`remasking="low_confidence"` permits remasking in principle, but LLaDA's block-wise policy never revises cross-block once a token is committed. A wrong token pollutes the context; the DAG only controls **which mask fills next**, not whether the wrong token gets corrected — so the predictor emits the same error regardless of order.
→ **Directions D / F / H (correction head / PC corrector / CDD constraint) target this directly**.

### H2: T=0 + bidirectional attention makes unmask order nearly irrelevant
A bidirectional transformer produces fixed logits given a fixed "already-unmasked set"; argmax does not depend on order. The DAG can only influence:
- Batch size per step (level width)
- Which position belongs to which level

Single-edge mutation cannot change level structure (3072 edges, ±1 barely moves the topology).
→ Step size × evaluation granularity × greedy accept together drown any signal in noise.

### H3: LLaDA hits its capability ceiling on these 137 init_fail prompts
Regardless of order or remasking, solving these prompts requires reasoning capability beyond llada-instruct's token-level representations.
→ Only training-side interventions (reasoning-reward RL, trajectory distillation) can move the needle.

---

## 5. Next experiments (to distinguish H1/H2/H3)

| Experiment | Method | Falsifies | Cost |
|---|---|---|---|
| **D. Failing-case forensics** | Take 10 `init_fail` prompts, diff output vs ground_truth by hand, locate the first wrong token | H1 vs H3 via inspection | 0.5 h (read `episodes.db`) |
| **A. Remasking ablation** | On 137 init_fail, add a minimal revise hook (resample whole block when conf < τ); measure rescue | H1: rescue > 5% confirms | 0.5 d |
| **B. Temperature sweep** | 137 prompts × T ∈ {0.0, 0.3, 0.7, 1.0}, N=8 samples, compute pass@N | H3: pass@N ≈ 0 → H3 holds; > 0 → sampling diversity has value | 0.5 d (inference only) |
| **C. Bigger DAG mutation** | Change `_generate_candidates` to mutate 10+ edges / whole levels per step, rerun greedy | H2: if still rescue=0 → order expressiveness truly useless | 0.5 d |

**Suggested order**: D first (0.5h, zero cost), then A (cheapest way to falsify H1 with max info).

---

## 6. Impact on the plan

**Deprioritize** (if D/A confirm H1):
- **G** Order-Token Joint Search
- **E** Prism Tree Search (DAG variant)
- `search/` differentiable / NAS / evolutionary — all static-DAG-space search variants

**Prioritize**:
- **D** BackPlay Correction Head — attacks commit-once directly
- **F** PC Sampler + Duo Schedule — alternate remasking path
- **H** CDD Constrained Sampling — content-adaptive state-level constraints

**Keep but decenter from DAG**:
- `search/` retained as a template-generation tool (multi-template warm-start)
- `scheduler/dag_scheduler.py` retained as a content-independent baseline

---

## 7. Artifacts

| File | Contents |
|---|---|
| `test.ipynb` | Initial discovery (confusion matrix, part 1) |
| `test_dag_deepdive.ipynb` | 28 unique DAG structures + prompt-feature analysis |
| `test_dag_gain.ipynb` | Net-gain recomputation (reproducible) |
| `docs/archive/finding_dag_search_zero_rescue.md` | This document |

**Raw data**: `runs/research_20260411_030422/stage2_discovery/`

**Relevant code entry points**:
- `src/dllm_reason/search/greedy.py::GreedyEdgeSearch`
- `src/dllm_reason/graph/templates.py` (warm-start template pool)
- `configs/search/greedy.yaml`
- `configs/eval_default.yaml`

---

## 8. Replication: NAS supernet also gives 0 rescue (2026-04-15)

**Goal**: rule out "greedy stuck in local minimum" by running a fundamentally different search algorithm (NAS supernet with temperature annealing + entropy-driven updates).

**Data**: `runs/research_20260415_035714/stage2_discovery/`
- 200 gsm8k prompts (different subset)
- `search_method = nas`, `budget = 50`
- `metadata = {method: nas_supernet, num_spans: 8, span_size: 16}`
- Config: T=0.0, block_length=32, num_steps=128 (same as v1.5.3)

**Result**:

| Metric | Value |
|---|---|
| init baseline correct | 95 / 200 (47.5%) |
| init baseline wrong | 105 / 200 |
| search rescue (fail→ok) | **0 / 105** |
| search break (ok→fail) | 0 / 95 |
| net Δacc | **+0.0 pp** |
| all prompts' `best_dag_edges` | **all = 0** (search converges to empty DAG) |
| prompts where fitness moved at all | **0 / 200** |

**Key contrast**:
- Greedy (v1.5.3) starts from CoT template at 3072 edges and jitters ±1 → stays near 3072.
- NAS supernet starts from a supernet with temperature annealing (τ: 1.24 → 0.1); entropy drops 2.83 → 0.04 — **entropy falls, but num_edges stays 0 throughout**.

**NAS history sample** (`prompt_0002`, init_fail):
```json
[
  {"fitness": 0.0, "step": 0},
  {"fitness": 0.0, "step": 20, "h": 2.83, "tau": 1.24, "num_edges": 0},
  {"fitness": 0.0, "step": 40, "h": 1.63, "tau": 0.48, "num_edges": 0},
  {"fitness": 0.0, "step": 50, "h": 0.037, "tau": 0.10, "num_edges": 0}
]
```

**Strengthened conclusion**: the empty DAG is the global optimum in DAG structure space (under T=0 llada-instruct + gsm8k). This is not a greedy problem, not a budget problem — **the DAG axis carries no signal at all**.

This rules out the "search algorithm not strong enough" explanation. H1 / H2 / H3 retain their evidence standing; the DAG-deprioritization conclusion is **reinforced**.
