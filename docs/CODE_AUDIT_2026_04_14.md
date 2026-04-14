# Code Audit — 2026-04-14

Verified findings from a manual review of dLLM-Reason on branch `dev` (commit `395ce92`).
An earlier automated scan produced many false positives; this document records only
claims I verified against the current source.

## Summary

- **Test suite:** 101/101 passing after fixes below.
- **2 real bugs fixed** in this pass (MDLM noise schedule, test sampling config).
- **5 earlier "critical" claims refuted** (KL sign, controller shape, transitive
  closure, STE formula, in-place `.zero_()`).
- **Two narrower questions remain worth investigating** (see bottom).

---

## Fixed in this audit

### 1. MDLM geometric noise schedule inverted — FIXED

**File:** `src/dllm_reason/models/mdlm.py:77-84, 91-94`

**Symptom:** `tests/test_models.py::TestMDLM::test_loss_scalar` asserted `loss > 0`
but observed `loss == 0.0` exactly.

**Root cause:** `sigma(t) = 1 - (1 - sigma_min)^t` with `sigma_min=1e-4` gives
`sigma(0) = 0` (correct) but `sigma(1) = 1e-4` (wrong — should be ≈1 for full
noise). Expected #masked tokens over B=2, L=16 was ~0.003, so `is_masked` was
almost always all-False, zeroing the loss.

**Fix:** `sigma(t) = 1 - sigma_min^t`, giving `sigma(1) = 1 - sigma_min ≈ 1`.
Derivative updated to match: `dsigma/dt = -sigma_min^t * log(sigma_min)`.

### 2. Sampling tests exceed `max_seq_len` — FIXED

**File:** `tests/test_models.py::TestSampling`

**Symptom:** `IndexError: index out of range in self` inside
`position_embedding` during both sampling tests.

**Root cause:** `SamplingConfig` defaults `block_length=32`, but the test model
is built with `max_seq_len=16`. The sampler auto-pads `gen_length` 16 → 32 to
satisfy divisibility by `block_length`, pushing positions beyond the embedding.

**Fix:** Tests now pass `block_length=SEQ_LEN` explicitly.

**Note:** This is a test-config bug, not a library bug — but the sampler could
reasonably warn/error when `prompt_len + padded_gen_length > model.max_seq_len`
instead of failing deep in the embedding lookup. Worth a follow-up.

---

## Refuted (earlier "critical" claims that do not hold)

These were flagged by an automated reviewer; I read the source and they are not
actually bugs. Recording here so they aren't re-raised.

### R1. KL sign in DiFFPO — NOT A BUG

**Location:** `src/dllm_reason/training/rl_train.py:468`

Claim: `kl_loss = cfg.kl_coeff * log_ratio.mean()` has wrong sign.

Verification: `log_ratio = policy_lp - ref_lp`. For samples drawn from the
policy, `E_policy[log_ratio] ≈ KL(policy ‖ ref) ≥ 0`. Adding
`+kl_coeff * log_ratio.mean()` to the loss penalizes divergence. Sign is
correct.

Caveat (quality, not correctness): the Schulman k3 estimator
`(ratio - 1) - log_ratio` is non-negative per-sample and lower variance.
Consider switching if KL estimates look noisy in practice.

### R2. DiFFPO controller embedding dim — NOT A CRASH

**Location:** `src/dllm_reason/training/rl_train.py:373, 402-403`

Claim: `emb = out.logits.mean(dim=1)` has shape `(B, V)` but controller
expects `hidden_dim`, crashing at the controller forward pass.

Verification: `_get_or_build_controller(hidden_dim)` is called with
`prompt_emb.shape[-1]`, which is `V`. The controller is built with matching
input size. Shapes are consistent — no crash. The inline comment at line 371
even acknowledges it's a proxy embedding. This is a modeling choice, not a
bug.

### R3. Transitive closure loop too short — NOT A BUG

**Location:** `src/dllm_reason/graph/dag.py:303-313`

Claim: `for _ in range(self._seq_len.bit_length())` is O(log n), but
closure "requires O(n) iterations."

Verification: the loop does matrix squaring (`reach = reach @ reach`), not
ordinary multiplication. After `k` squarings, `reach = (I + A)^(2^k)`. Since
`2^bit_length(n) ≥ n`, all reachability paths are captured. O(log n)
squarings is correct and is the standard fast closure algorithm.

### R4. Straight-through estimator broken — NOT A BUG

**Location:** `src/dllm_reason/search/differentiable.py:181`

Claim: `(hard - probs).detach() + probs` is wrong; "correct" form is
`hard - probs.detach() + probs`.

Verification:
- Forward value: `(H - P) + P = H` ✓
- Backward: the detached term contributes zero gradient; only the trailing
  `+ probs` contributes, yielding `∂P/∂θ`.

The two forms are mathematically equivalent.

### R5. In-place `.zero_()` on grad tensor — NOT CRITICAL

**Location:** `src/dllm_reason/scheduler/adaptive_dynamic_scheduler.py:150`

Claim: autograd corruption in E2E DAG learning.

Verification: the adaptive dynamic scheduler is invoked inside
`DiffusionSampler.sample`, which is decorated with `@torch.no_grad()`
(sampler.py:100). No gradient context, no autograd corruption. If someone
later wires this scheduler into a differentiable training path, this would
need revisiting — but it is not a current bug.

---

## Narrower questions worth deeper investigation

### Q1. Is DiFFPO's KL estimator numerically reasonable?

Current code uses the plain `log_ratio.mean()` estimator, which:
- Is an unbiased estimator of KL only when samples come from the policy
  (verified here — `seqs` is sampled from the current policy inside the group
  loop).
- Can go negative in any finite sample and has high variance.
- The Schulman k3 estimator `(ratio - 1) - log_ratio` is non-negative and
  lower variance.

**Action needed:** log per-iteration KL during a real training run. If it
swings negative often or its variance dominates `ppo_loss`, switch to k3.

### Q2. Does `adaptive_dynamic` always produce valid topological orders?

The scheduler builds a *soft* influence graph at runtime and selects positions
by readiness fraction. It does not explicitly guarantee that a strict
topological order is respected — it is a soft heuristic. Open questions:

- When `influence_threshold` is low, can the scheduler unmask a "child" before
  its soft parents are all committed? (From `_compute_readiness`: yes — a
  position with readiness `parent_unmasked / parent_count < 1` is still
  eligible if it has the highest readiness among masked positions.)
- Is there a cycle-detection step? (No.)
- How does this interact with DAG-guided evaluation metrics that assume a
  strict order?

**Action needed:** instrument a run on GSM8K and record, per step, how often
the chosen position has unsatisfied soft parents. If frequent, the strategy
name "DAG" is misleading and should be documented as a soft influence
heuristic.

### Q3. Sampler silently oversizes past `max_seq_len`

**Location:** `src/dllm_reason/inference/sampler.py:120-138`

The auto-pad logic can extend `prompt_len + gen_length` beyond the backbone's
`max_seq_len`, triggering an opaque `IndexError` in the position embedding
(this is what broke both sampling tests before I changed the test config).

**Action needed:** add an assertion in `DiffusionSampler.sample` that
`prompt_len + padded_gen_length <= self.model.max_seq_len` with a clear error
message.

### Q4. `DAGStore` SQLite connection lifecycle

Not verified yet — worth checking whether `__del__` / context-manager
semantics are correct under concurrent `collect_episodes` → `learn_from_episodes`
pipelines. Lock contention would manifest as "database is locked" errors under
load, not at test time.

---

## Test results

```
$ pytest tests/ -q
101 passed, 3 warnings in 8.75s
```

All test files: `test_dag.py`, `test_library.py`, `test_models.py`,
`test_schedulers.py`, `test_search_levels.py`, `test_training_modules.py`.
