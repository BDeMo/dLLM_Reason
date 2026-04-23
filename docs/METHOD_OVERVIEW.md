# dLLM-Reason — Method Overview

> **Seven-sentence summary with file-level evidence.**
> Every claim is linked to the source that supports it.

---

## Core Argument

Discrete diffusion language models (dLLMs) denoise an entire fixed-length canvas in
parallel, so the order in which masked positions are revealed is a free degree of
freedom that autoregressive models do not have.
dLLM-Reason exploits that degree of freedom by imposing a **Directed Acyclic Graph
(DAG) over token positions**: an edge *i → j* means position *i* must be unmasked
before position *j*, encoding reasoning dependencies directly into the sampling loop
without touching model weights.

---

## The Seven Facts

**1. The reasoning gap is a canvas constraint problem.**
Unlike autoregressive models whose output length is emergent, dLLMs write into a
pre-allocated canvas of fixed size ([`SamplingConfig.gen_length`](../src/dllm_reason/inference/sampler.py)).
This creates a hard structural constraint that governs what the model can express —
and is therefore a lever to pull on.

**2. The DAG is a GPU-native boolean adjacency matrix.**
[`TokenDAG`](../src/dllm_reason/graph/dag.py) stores dependencies as a `(L, L)` bool
tensor on device.
`ready_positions(is_unmasked)` is a single batched matrix operation
(`(~adj).unsqueeze(0) | is_unmasked.unsqueeze(-1)` reduced with `.all(dim=1)`),
making topological scheduling cost-free relative to the LM forward pass.

**3. Thirteen unmasking schedulers explore the strategy space.**
Eight flat schedulers (confidence, entropy, random, semi-AR, maskgit-cosine,
critical-token-first, curriculum, linear) share one interface with four DAG-guided
schedulers (cot, skeleton, bidirectional, answer-first) and one dynamic scheduler
that constructs a soft influence graph at runtime
(threshold = 0.3, momentum = 0.5).
All are registered under
[`src/dllm_reason/inference/schedulers/`](../src/dllm_reason/inference/schedulers/).

**4. Empirical hypothesis testing on 60 GSM8K fail-prompts reveals a granularity ladder.**
Fine-grained interventions — token-level revision (H1), unmask-order DAG search (H2),
span-level revise (A3) — all achieve **rescue rate = 0.00%**.
Coarse-grained interventions do succeed: block-layout variations (A4) rescue 8.33%,
prompt-template scaffolding (A5) rescues 13.33%, and generation-length sweep (A6)
rescues 20.00%.
The union ceiling across all methods reaches 91.67% (55/60 prompts), confirming that
five prompts are hard capacity failures independent of strategy.
Results are logged in [`docs/archive/hypotheses.md`](../docs/archive/hypotheses.md) and
the per-axis findings under [`docs/archive/`](../docs/archive/).

**5. Training is aligned with DAG-guided inference via topology-biased masking.**
[`DAGAwareTrainer`](../src/dllm_reason/training/dag_aware_train.py) raises the masking
probability for tokens at higher topological levels
(`level_bias = level_idx / max_levels`), so the model sees the same unmasking order
at training time that the DAG scheduler enforces at inference time — closing the
train-inference distribution gap that standard random masking leaves open
(cf. PUMA, [`progressive_train.py`](../src/dllm_reason/training/progressive_train.py)).

**6. A six-level search taxonomy spans eight orders of magnitude in search space.**
Starting from template enumeration (8 candidates) through greedy perturbation (~10²),
evolutionary search (population = 20, mutation = 0.3, crossover = 0.5, ~10³),
RL-policy construction (GRU + REINFORCE, ~10⁴), NOTEARS continuous relaxation (ℝ^n²),
DARTS/ENAS NAS supernet (span_size = 16), and joint end-to-end optimisation
(lr_dag = 3e-3, sparsity = 0.01).
All six are implemented in
[`src/dllm_reason/search/`](../src/dllm_reason/search/) and share the
`eval_fn(model, dag) → float` interface.

**7. v1.6 closes the loop with canvas-structured teacher distillation.**
An AR teacher (Qwen3/3.5) generates structured traces — `<SETUP>`, `<STEP_x>`,
`<ANSWER>` — on 2 000 GSM8K training prompts; five retries at temperature 0.7 produce
4 000–6 000 diverse valid rows.
LLaDA-8B is then fine-tuned with masked-diffusion SFT restricted to the answer region,
and the entire pipeline (download → SFT → eval → ablate → pass\@N) runs end-to-end
via [`scripts/validate/t6t7_train.py`](../scripts/validate/t6t7_train.py) and
[`scripts/t6_ablate.sh`](../scripts/t6_ablate.sh).
Release notes: [`docs/V1.6.1_RELEASE.md`](../docs/V1.6.1_RELEASE.md).

---

## Architecture Diagram (text)

```
Prompt
  │
  ▼
┌──────────────────────────────────────────────┐
│  DAG Construction                            │
│  templates / search (L0–L6)                  │
│  src/dllm_reason/graph/  + search/           │
└────────────────┬─────────────────────────────┘
                 │ TokenDAG (L×L bool)
                 ▼
┌──────────────────────────────────────────────┐
│  DiffusionSampler                            │
│  block_length=32, num_steps=128, T=0         │
│  UnmaskingScheduler (13 variants)            │
│  src/dllm_reason/inference/sampler.py        │
└────────────────┬─────────────────────────────┘
                 │ episode (correct?, score)
                 ▼
┌──────────────────────────────────────────────┐
│  EpisodeStore (SQLite)                       │
│  src/dllm_reason/library/episode.py          │
└────────┬───────────────────┬─────────────────┘
         │ SFT pairs         │ GRPO / DiFFPO
         ▼                   ▼
┌──────────────┐   ┌─────────────────────────┐
│ T6/T7 SFT   │   │ RL trainers             │
│ (LoRA)      │   │ DiffuGRPO / DiFFPO /    │
│             │   │ UnmaskingPolicyRL        │
└─────────────┘   └─────────────────────────┘
```

---

## Key Numbers at a Glance

| Quantity | Value | Source |
|---|---|---|
| Canvas size (canonical) | 128 tokens | [`SamplingConfig`](../src/dllm_reason/inference/sampler.py) |
| Block length | 32 | [`SamplingConfig`](../src/dllm_reason/inference/sampler.py) |
| Denoising steps | 128 | [`SamplingConfig`](../src/dllm_reason/inference/sampler.py) |
| Unmasking schedulers | 13 | [`schedulers/`](../src/dllm_reason/inference/schedulers/) |
| DAG search levels | 6 (L0–L5) | [`search/`](../src/dllm_reason/search/) |
| Strategy-search space | 384 configs / prompt | [`scripts/strategy_search.py`](../scripts/strategy_search.py) |
| GSM8K fail-prompt test set | 60 prompts | [`docs/archive/hypotheses.md`](../docs/archive/hypotheses.md) |
| Best single-axis rescue (A6) | 20.00% | [`docs/archive/hypotheses.md`](../docs/archive/hypotheses.md) |
| Union rescue ceiling | 91.67% (55/60) | [`docs/archive/hypotheses.md`](../docs/archive/hypotheses.md) |
| Hard capacity failures | 5 prompts {4,5,14,41,42} | [`docs/archive/hypotheses.md`](../docs/archive/hypotheses.md) |
| T6 SFT teacher | Qwen3/3.5-instruct | [`scripts/validate/t6t7_train.py`](../scripts/validate/t6t7_train.py) |
| T6 SFT rows produced | 4 000–6 000 | [`docs/V1.6.1_RELEASE.md`](../docs/V1.6.1_RELEASE.md) |
| Primary eval model | LLaDA-8B-Instruct | [`src/dllm_reason/models/llada.py`](../src/dllm_reason/models/llada.py) |
