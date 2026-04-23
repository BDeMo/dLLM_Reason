"""Inspect Stage 2 DAG discovery results.

Shows per-prompt best DAG: which template won, what the model generated,
topological level structure, and the DAG's visual representation.

Usage:
    # Default: read from latest run directory
    python scripts/inspect_stage2.py --run_dir runs/research_20260410

    # Filter by dataset or strategy
    python scripts/inspect_stage2.py --run_dir runs/research_20260410 \
        --dataset gsm8k --strategy cot

    # Show only the first N prompts
    python scripts/inspect_stage2.py --run_dir runs/research_20260410 -n 5

    # Show full DAG adjacency (instead of summary)
    python scripts/inspect_stage2.py --run_dir runs/research_20260410 --full_dag

    # Export to JSON for further analysis
    python scripts/inspect_stage2.py --run_dir runs/research_20260410 --export out.json

    # Summary only (no per-prompt details)
    python scripts/inspect_stage2.py --run_dir runs/research_20260410 --summary
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect Stage 2 DAG discovery results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run_dir", type=str, required=True,
                    help="Pipeline run directory (contains stage2_discovery/)")
    p.add_argument("--dataset", type=str, default=None,
                    help="Filter by dataset (e.g. gsm8k)")
    p.add_argument("--strategy", type=str, default=None,
                    help="Filter by best strategy (e.g. cot, skeleton)")
    p.add_argument("-n", "--num", type=int, default=None,
                    help="Show at most N prompts")
    p.add_argument("--full_dag", action="store_true",
                    help="Show full DAG visual for each prompt")
    p.add_argument("--export", type=str, default=None,
                    help="Export detailed results to JSON file")
    p.add_argument("--summary", action="store_true",
                    help="Show summary statistics only")
    p.add_argument("--prompt", type=str, default=None,
                    help="Search for a specific prompt (substring match)")
    return p.parse_args()


def load_best_map(run_dir: Path) -> dict:
    """Load best_dag_per_prompt.json from stage2_discovery/."""
    path = run_dir / "stage2_discovery" / "best_dag_per_prompt.json"
    if not path.exists():
        print(f"Error: {path} not found.")
        print("Make sure --run_dir points to a pipeline run with Stage 2 completed.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_episodes_for_prompt(db_path: str, prompt: str) -> list:
    """Load all episodes for a given prompt from the episode store."""
    try:
        from dllm_reason.library.episode import EpisodeStore  # fixed: was dllm_reason.episodes.store
        store = EpisodeStore(db_path)
        all_eps = store.query(limit=100000)
        return [ep for ep in all_eps if ep.prompt == prompt]
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import EpisodeStore: {e}. Episode details will be skipped.")
        return []
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load episodes from {db_path}: {e}")
        return []


def build_dag_from_episode(ep) -> "TokenDAG | None":
    """Reconstruct a TokenDAG from an episode's dag_adjacency."""
    if not ep.dag_adjacency or not ep.dag_seq_len:
        return None
    try:
        from dllm_reason.graph.dag import TokenDAG
        flat = [cell for row in ep.dag_adjacency for cell in row]
        adj = torch.tensor(flat, dtype=torch.bool).reshape(
            ep.dag_seq_len, ep.dag_seq_len)
        return TokenDAG(adj)
    except Exception:
        return None


def build_dag_from_template(strategy: str, seq_len: int = 128) -> "TokenDAG | None":
    """Build a DAG from a template name."""
    try:
        from dllm_reason.graph.templates import build_template
        return build_template(strategy, seq_len, device="cpu")
    except Exception:
        return None


def dag_summary(dag) -> str:
    """One-line summary of a DAG."""
    levels = dag.topological_levels()
    level_sizes = [len(lv) for lv in levels]
    return (f"edges={dag.num_edges()}, depth={len(levels)}, "
            f"level_sizes={level_sizes}")


def dag_level_detail(dag) -> str:
    """Multi-line view of topological levels with position ranges."""
    levels = dag.topological_levels()
    lines = []
    for i, positions in enumerate(levels):
        positions_sorted = sorted(positions)
        # Show as ranges for readability
        ranges = _compress_ranges(positions_sorted)
        lines.append(f"  Level {i:>2}: {len(positions):>4} positions  {ranges}")
    return "\n".join(lines)


def _compress_ranges(nums: list[int]) -> str:
    """Compress [0,1,2,3,5,6,8] into '0-3, 5-6, 8'."""
    if not nums:
        return "[]"
    ranges = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    # Truncate if too long
    text = ", ".join(ranges)
    if len(text) > 120:
        text = text[:117] + "..."
    return f"[{text}]"


def print_prompt_detail(idx: int, prompt: str, info: dict, episodes: list,
                        show_full_dag: bool = False):
    """Print detailed info for one prompt."""
    strategy = info.get("best_strategy") or "(none)"
    correct = info.get("correct", False)
    task = info.get("task_type", "?")
    n_tried = info.get("num_strategies_tried", 0)
    n_correct = info.get("num_correct", 0)
    output = info.get("output", "")
    gt = info.get("ground_truth", "")

    print(f"\n{'━' * 80}")
    print(f"  #{idx+1}  [{task}]  best={strategy}  "
          f"correct={correct}  tried={n_tried}  correct_count={n_correct}")
    print(f"{'━' * 80}")

    # Prompt (truncated)
    prompt_display = prompt[:200] + "..." if len(prompt) > 200 else prompt
    print(f"  Prompt:  {prompt_display}")
    print(f"  GT:      {gt[:100]}")
    if output:
        print(f"  Output:  {output[:200]}{'...' if len(output) > 200 else ''}")

    # Per-strategy breakdown
    if episodes:
        print(f"\n  Strategy results ({len(episodes)} episodes):")
        by_strat = defaultdict(list)
        for ep in episodes:
            by_strat[ep.strategy_name].append(ep)
        for sname, eps in sorted(by_strat.items()):
            n_ok = sum(1 for e in eps if e.correct)
            scores = [e.score for e in eps if e.score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0
            marker = " ◀ best" if sname == strategy else ""
            print(f"    {sname:<20} correct={n_ok}/{len(eps)}  "
                  f"avg_score={avg_score:.3f}{marker}")

    # DAG visualization
    dag = None
    # First try to get DAG from episode adjacency
    if episodes:
        best_eps = [e for e in episodes if e.strategy_name == strategy]
        for ep in best_eps:
            dag = build_dag_from_episode(ep)
            if dag:
                break

    # Fallback: build from template name
    if dag is None and strategy and strategy != "(none)":
        dag = build_dag_from_template(strategy)

    if dag:
        print(f"\n  DAG structure ({strategy}):")
        print(f"    {dag_summary(dag)}")
        print()
        print(dag_level_detail(dag))
        if show_full_dag:
            print(f"\n  DAG adjacency visual:")
            for line in str(dag).split("\n"):
                print(f"    {line}")


def print_summary(best_map: dict):
    """Print aggregate statistics."""
    total = len(best_map)
    has_best = sum(1 for v in best_map.values() if v.get("best_strategy"))
    no_answer = total - has_best

    by_strategy = defaultdict(int)
    by_task = defaultdict(lambda: {"total": 0, "correct": 0})
    for v in best_map.values():
        s = v.get("best_strategy")
        if s:
            by_strategy[s] += 1
        task = v.get("task_type", "unknown")
        by_task[task]["total"] += 1
        if v.get("correct"):
            by_task[task]["correct"] += 1

    print(f"\n{'=' * 60}")
    print(f"  Stage 2 — Discovery Summary")
    print(f"{'=' * 60}")
    print(f"  Total prompts:      {total}")
    print(f"  With best DAG:      {has_best}  ({has_best/max(total,1)*100:.1f}%)")
    print(f"  No correct answer:  {no_answer}")

    print(f"\n  By strategy:")
    for s, cnt in sorted(by_strategy.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / max(total, 1) * 40)
        print(f"    {s:<25} {cnt:>4}  ({cnt/max(total,1)*100:5.1f}%)  {bar}")

    print(f"\n  By dataset:")
    for task, stats in sorted(by_task.items()):
        acc = stats["correct"] / max(stats["total"], 1) * 100
        print(f"    {task:<15} {stats['correct']:>4}/{stats['total']:<4}  ({acc:.1f}%)")

    # Template DAG structure reference
    print(f"\n  Template DAG structures (seq_len=128):")
    try:
        from dllm_reason.graph.templates import TEMPLATE_NAMES, build_template
        for tname in TEMPLATE_NAMES:
            try:
                dag = build_template(tname, 128, "cpu")
                print(f"    {tname:<20} {dag_summary(dag)}")
            except Exception:
                pass
    except ImportError:
        pass
    print()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    best_map = load_best_map(run_dir)

    # Apply filters
    items = list(best_map.values())
    if args.dataset:
        items = [v for v in items if v.get("task_type") == args.dataset]
    if args.strategy:
        items = [v for v in items if v.get("best_strategy") == args.strategy]
    if args.prompt:
        items = [v for v in items
                 if args.prompt.lower() in v.get("prompt", "").lower()]

    if not items:
        print("No matching prompts found.")
        if args.dataset:
            print(f"  (filtered by dataset={args.dataset})")
        if args.strategy:
            print(f"  (filtered by strategy={args.strategy})")
        return

    # Summary
    print_summary(best_map if not (args.dataset or args.strategy) else
                  {v["prompt"]: v for v in items})

    if args.summary:
        return

    # Per-prompt detail
    if args.num:
        items = items[:args.num]

    db_path = str(run_dir / "stage2_discovery" / "episodes.db")
    print(f"\nShowing {len(items)} prompt(s):")

    for idx, info in enumerate(items):
        prompt = info.get("prompt", "")
        episodes = load_episodes_for_prompt(db_path, prompt)
        print_prompt_detail(idx, prompt, info, episodes,
                            show_full_dag=args.full_dag)

    # Export
    if args.export:
        export_data = []
        for info in items:
            prompt = info.get("prompt", "")
            strategy = info.get("best_strategy")
            dag = None
            if strategy:
                dag = build_dag_from_template(strategy)
            entry = {
                **info,
                "dag_info": dag_summary(dag) if dag else None,
                "dag_levels": None,
            }
            if dag:
                levels = dag.topological_levels()
                entry["dag_levels"] = [sorted(lv) for lv in levels]
            export_data.append(entry)

        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\nExported {len(export_data)} entries to {out_path}")


if __name__ == "__main__":
    main()
