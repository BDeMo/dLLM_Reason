"""Test all DAG search levels (L0–L6) with mock model and eval_fn.

No GPU or real model required. Each level is tested for:
1. Correct instantiation (accepts documented CLI params)
2. search() runs to completion within budget
3. Returns valid SearchResult with correct fields
4. best_dag is a valid TokenDAG with correct seq_len
"""

from __future__ import annotations

import pytest
import torch

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.search.base import SearchResult

# ── Shared fixtures ──────────────────────────────────────────────────────────

SEQ_LEN = 16  # Small for fast tests


class MockModel:
    """Minimal mock satisfying DAGSearcher.search(model=...)."""

    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        return "42"

    def __call__(self, *args, **kwargs):
        return None


class MockModelWithParameters(MockModel):
    """Model mock that also exposes .parameters() for gradient-based methods."""

    def parameters(self):
        return [torch.zeros(1, requires_grad=True)]


def simple_eval_fn(model, dag: TokenDAG) -> float:
    """Deterministic eval_fn: more edges = higher fitness (capped at 1.0)."""
    return min(dag.num_edges() / max(dag.seq_len, 1), 1.0)


def random_eval_fn(model, dag: TokenDAG) -> float:
    """Stochastic eval_fn for testing variance handling."""
    return torch.rand(1).item()


def _assert_valid_result(result: SearchResult, seq_len: int, method: str):
    """Validate SearchResult structure."""
    assert isinstance(result, SearchResult), f"Expected SearchResult, got {type(result)}"
    assert isinstance(result.best_dag, TokenDAG), "best_dag is not a TokenDAG"
    assert result.best_dag.seq_len == seq_len, (
        f"DAG seq_len={result.best_dag.seq_len}, expected {seq_len}"
    )
    assert isinstance(result.best_fitness, (int, float)), "best_fitness not numeric"
    assert isinstance(result.history, list), "history not a list"
    assert isinstance(result.metadata, dict), "metadata not a dict"
    assert result.best_dag.is_valid(), f"best_dag from {method} is not a valid DAG"


# ── Import helpers (trigger registration) ────────────────────────────────────

@pytest.fixture(autouse=True)
def register_all_searchers():
    """Import all search modules so they register in SEARCH_REGISTRY."""
    import dllm_reason.search.greedy           # noqa: F401
    import dllm_reason.search.evolutionary     # noqa: F401
    import dllm_reason.search.rl_policy        # noqa: F401
    import dllm_reason.search.differentiable   # noqa: F401
    import dllm_reason.search.nas_search       # noqa: F401
    import dllm_reason.search.e2e_dag_learner  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
#  L0: Enumerate — Template sweep
# ═══════════════════════════════════════════════════════════════════════════════

class TestL0Enumerate:
    """L0: build_all_templates and evaluate each one."""

    def test_template_sweep(self):
        from dllm_reason.graph.templates import build_all_templates, TEMPLATE_NAMES

        templates = build_all_templates(SEQ_LEN, device="cpu")
        assert len(templates) > 0, "No templates built"

        model = MockModel()
        best_dag, best_fit = None, -float("inf")
        for name, dag in templates.items():
            assert isinstance(dag, TokenDAG)
            assert dag.seq_len == SEQ_LEN
            assert dag.is_valid(), f"Template {name} is not a valid DAG"
            fit = simple_eval_fn(model, dag)
            if fit > best_fit:
                best_dag, best_fit = dag, fit

        assert best_dag is not None
        assert best_fit >= 0.0

    def test_all_template_names_build(self):
        from dllm_reason.graph.templates import TEMPLATE_NAMES, build_template

        for name in TEMPLATE_NAMES:
            dag = build_template(name, SEQ_LEN, device="cpu")
            assert isinstance(dag, TokenDAG)
            assert dag.seq_len == SEQ_LEN
            assert dag.is_valid(), f"Template {name} produced invalid DAG"


# ═══════════════════════════════════════════════════════════════════════════════
#  L1: Perturb — Greedy edge search
# ═══════════════════════════════════════════════════════════════════════════════

class TestL1Perturb:
    """L1: GreedyEdgeSearch."""

    def test_greedy_basic(self):
        from dllm_reason.search.greedy import GreedyEdgeSearch

        searcher = GreedyEdgeSearch(
            num_candidates=5,
            patience=3,
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=20,
        )
        _assert_valid_result(result, SEQ_LEN, "greedy")
        assert result.metadata.get("method") == "greedy"

    def test_greedy_with_template_warmstart(self):
        from dllm_reason.search.greedy import GreedyEdgeSearch

        searcher = GreedyEdgeSearch(
            init_templates=["cot", "skeleton"],
            num_candidates=3,
            patience=2,
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=20,
        )
        _assert_valid_result(result, SEQ_LEN, "greedy+templates")
        # Should have used at least 2 evals for template warm-start
        assert len(result.history) >= 2

    def test_greedy_with_initial_dag(self):
        from dllm_reason.search.greedy import GreedyEdgeSearch

        init_dag = TokenDAG.no_edges(SEQ_LEN, device="cpu")
        searcher = GreedyEdgeSearch(
            initial_dag=init_dag,
            num_candidates=5,
            patience=2,
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=15,
        )
        _assert_valid_result(result, SEQ_LEN, "greedy+init_dag")

    def test_greedy_cli_params(self):
        """Test with the exact CLI param names from run_research_pipeline.py."""
        from dllm_reason.search.greedy import GreedyEdgeSearch
        from dllm_reason.graph.templates import TEMPLATE_NAMES

        searcher = GreedyEdgeSearch(
            init_templates=list(TEMPLATE_NAMES),
            num_candidates=10,   # --s2_greedy_candidates
            patience=5,          # --s2_greedy_patience
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=random_eval_fn,
            seq_len=SEQ_LEN,
            budget=30,
        )
        _assert_valid_result(result, SEQ_LEN, "greedy+cli")


# ═══════════════════════════════════════════════════════════════════════════════
#  L2: Evolve — Evolutionary search
# ═══════════════════════════════════════════════════════════════════════════════

class TestL2Evolve:
    """L2: EvolutionarySearch."""

    def test_evolutionary_basic(self):
        from dllm_reason.search.evolutionary import EvolutionarySearch

        searcher = EvolutionarySearch(
            population_size=6,
            mutation_rate=0.3,
            crossover_rate=0.5,
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=20,
        )
        _assert_valid_result(result, SEQ_LEN, "evolutionary")
        assert result.metadata.get("method") == "evolutionary"
        assert result.metadata.get("generations", 0) > 0

    def test_evolutionary_with_templates(self):
        from dllm_reason.search.evolutionary import EvolutionarySearch

        searcher = EvolutionarySearch(
            init_templates=["cot", "skeleton", "bidirectional"],
            population_size=6,
            mutation_rate=0.5,
            crossover_rate=0.5,
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=random_eval_fn,
            seq_len=SEQ_LEN,
            budget=20,
        )
        _assert_valid_result(result, SEQ_LEN, "evolutionary+templates")

    def test_evolutionary_cli_params(self):
        """Test with the exact CLI param names from run_research_pipeline.py."""
        from dllm_reason.search.evolutionary import EvolutionarySearch
        from dllm_reason.graph.templates import TEMPLATE_NAMES

        budget = 30
        searcher = EvolutionarySearch(
            init_templates=list(TEMPLATE_NAMES),
            population_size=min(20, budget // 2),  # --s2_evo_pop_size
            mutation_rate=0.3,                       # --s2_evo_mutation_rate
            crossover_rate=0.5,                      # --s2_evo_crossover_rate
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=random_eval_fn,
            seq_len=SEQ_LEN,
            budget=budget,
        )
        _assert_valid_result(result, SEQ_LEN, "evolutionary+cli")


# ═══════════════════════════════════════════════════════════════════════════════
#  L3: Construct — RL policy
# ═══════════════════════════════════════════════════════════════════════════════

class TestL3Construct:
    """L3: RLPolicySearch."""

    def test_rl_policy_basic(self):
        from dllm_reason.search.rl_policy import RLPolicySearch

        searcher = RLPolicySearch(
            max_seq_len=SEQ_LEN,
            hidden_dim=32,
            lr=1e-3,
            max_edges_per_dag=10,
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=10,
        )
        _assert_valid_result(result, SEQ_LEN, "rl_policy")

    def test_rl_policy_cli_params(self):
        """Test with the exact CLI param names from run_research_pipeline.py."""
        from dllm_reason.search.rl_policy import RLPolicySearch

        searcher = RLPolicySearch(
            max_seq_len=SEQ_LEN,           # args.gen_length
            hidden_dim=128,                 # --s2_rl_hidden_dim
            lr=1e-4,                        # --s2_rl_lr
            max_edges_per_dag=50,           # --s2_rl_max_edges
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=random_eval_fn,
            seq_len=SEQ_LEN,
            budget=8,
        )
        _assert_valid_result(result, SEQ_LEN, "rl_policy+cli")


# ═══════════════════════════════════════════════════════════════════════════════
#  L4: Relax — Differentiable (NOTEARS)
# ═══════════════════════════════════════════════════════════════════════════════

class TestL4Relax:
    """L4: DifferentiableDAGSearch."""

    def test_differentiable_basic(self):
        from dllm_reason.search.differentiable import DifferentiableDAGSearch

        searcher = DifferentiableDAGSearch(
            lr=1e-2,
            rho_init=1.0,
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=10,
        )
        _assert_valid_result(result, SEQ_LEN, "differentiable")

    def test_differentiable_cli_params(self):
        """Test with the exact CLI param names from run_research_pipeline.py."""
        from dllm_reason.search.differentiable import DifferentiableDAGSearch

        searcher = DifferentiableDAGSearch(
            lr=1e-3,           # --s2_diff_lr
            rho_init=1.0,      # --s2_diff_rho_init
        )
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=random_eval_fn,
            seq_len=SEQ_LEN,
            budget=8,
        )
        _assert_valid_result(result, SEQ_LEN, "differentiable+cli")

    def test_notears_acyclicity(self):
        """Verify the returned DAG is actually acyclic."""
        from dllm_reason.search.differentiable import DifferentiableDAGSearch

        searcher = DifferentiableDAGSearch(lr=1e-2, rho_init=1.0)
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=15,
        )
        assert result.best_dag.is_valid(), "NOTEARS-produced DAG has cycles"


# ═══════════════════════════════════════════════════════════════════════════════
#  L5: Architect — NAS search
# ═══════════════════════════════════════════════════════════════════════════════

class TestL5Architect:
    """L5: NASDAGSearch."""

    def test_nas_supernet(self):
        from dllm_reason.search.nas_search import NASDAGSearch, NASConfig

        config = NASConfig(
            mode="supernet",
            span_size=4,        # SEQ_LEN=16 / 4 = 4 spans
            proxy_samples=2,
            full_eval_every=5,
        )
        searcher = NASDAGSearch(config=config)
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=10,
        )
        _assert_valid_result(result, SEQ_LEN, "nas_supernet")

    def test_nas_controller(self):
        from dllm_reason.search.nas_search import NASDAGSearch, NASConfig

        config = NASConfig(
            mode="controller",
            span_size=4,
            controller_hidden=32,
            controller_batch=2,
            proxy_samples=2,
            full_eval_every=5,
        )
        searcher = NASDAGSearch(config=config)
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=10,
        )
        _assert_valid_result(result, SEQ_LEN, "nas_controller")

    def test_nas_cli_params(self):
        """Test with the exact CLI param names from run_research_pipeline.py."""
        from dllm_reason.search.nas_search import NASDAGSearch, NASConfig

        config = NASConfig(
            mode="supernet",       # --s2_nas_mode
            span_size=4,           # --s2_nas_span_size (using 4 for small SEQ_LEN)
        )
        searcher = NASDAGSearch(config=config)
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=random_eval_fn,
            seq_len=SEQ_LEN,
            budget=10,
        )
        _assert_valid_result(result, SEQ_LEN, "nas+cli")


# ═══════════════════════════════════════════════════════════════════════════════
#  L6: Learn — End-to-end joint optimization
# ═══════════════════════════════════════════════════════════════════════════════

class TestL6Learn:
    """L6: E2EDAGLearner."""

    def test_e2e_basic(self):
        from dllm_reason.search.e2e_dag_learner import E2EDAGLearner, E2EConfig

        config = E2EConfig(
            lr_dag=3e-2,
            warmup_steps=2,
            checkpoint_every=100,
        )
        searcher = E2EDAGLearner(config=config)
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=10,
        )
        _assert_valid_result(result, SEQ_LEN, "e2e")

    def test_e2e_cli_params(self):
        """Test with the exact CLI param names from run_research_pipeline.py."""
        from dllm_reason.search.e2e_dag_learner import E2EDAGLearner, E2EConfig

        config = E2EConfig(
            lr_dag=3e-3,                  # --s2_e2e_lr
            sparsity_weight=0.01,         # --s2_e2e_sparsity
        )
        searcher = E2EDAGLearner(config=config)
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=random_eval_fn,
            seq_len=SEQ_LEN,
            budget=8,
        )
        _assert_valid_result(result, SEQ_LEN, "e2e+cli")

    def test_e2e_with_init_dag(self):
        from dllm_reason.search.e2e_dag_learner import E2EDAGLearner, E2EConfig

        init_dag = TokenDAG.no_edges(SEQ_LEN, device="cpu")
        config = E2EConfig(lr_dag=3e-2, warmup_steps=1, checkpoint_every=100)
        searcher = E2EDAGLearner(config=config, init_dag=init_dag)
        model = MockModel()
        result = searcher.search(
            model=model,
            eval_fn=simple_eval_fn,
            seq_len=SEQ_LEN,
            budget=8,
        )
        _assert_valid_result(result, SEQ_LEN, "e2e+init_dag")


# ═══════════════════════════════════════════════════════════════════════════════
#  SEARCH_REGISTRY integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchRegistry:
    """Verify SEARCH_REGISTRY contains all expected methods."""

    def test_registry_has_all_methods(self):
        from dllm_reason.search import SEARCH_REGISTRY

        expected = {"greedy", "evolutionary", "rl_policy",
                    "differentiable", "nas", "e2e"}
        registered = set(SEARCH_REGISTRY.keys())
        missing = expected - registered
        assert not missing, f"Missing from SEARCH_REGISTRY: {missing}"

    def test_registry_instantiation_round_trip(self):
        """Instantiate each registered searcher and run minimal search."""
        from dllm_reason.search import SEARCH_REGISTRY

        model = MockModel()
        for name in SEARCH_REGISTRY.keys():
            cls = SEARCH_REGISTRY.get(name)
            # Minimal kwargs for each
            kwargs = _minimal_kwargs(name)
            searcher = cls(**kwargs)
            result = searcher.search(
                model=model,
                eval_fn=simple_eval_fn,
                seq_len=SEQ_LEN,
                budget=8,
            )
            _assert_valid_result(result, SEQ_LEN, f"registry:{name}")


def _minimal_kwargs(method: str) -> dict:
    """Return minimal __init__ kwargs for a searcher from the registry."""
    if method == "greedy":
        return {"num_candidates": 3, "patience": 2}
    elif method == "evolutionary":
        return {"population_size": 4, "mutation_rate": 0.3, "crossover_rate": 0.5}
    elif method == "rl_policy":
        return {"max_seq_len": SEQ_LEN, "hidden_dim": 32, "lr": 1e-3,
                "max_edges_per_dag": 10}
    elif method == "differentiable":
        return {"lr": 1e-2, "rho_init": 1.0}
    elif method == "nas":
        from dllm_reason.search.nas_search import NASConfig
        return {"config": NASConfig(mode="supernet", span_size=4,
                                     proxy_samples=2, full_eval_every=5)}
    elif method == "e2e":
        from dllm_reason.search.e2e_dag_learner import E2EConfig
        return {"config": E2EConfig(lr_dag=3e-2, warmup_steps=1,
                                     checkpoint_every=100)}
    else:
        return {}
