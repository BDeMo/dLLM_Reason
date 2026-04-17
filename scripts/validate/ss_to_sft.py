"""ss_to_sft.py — strategy_search winners.json → SFT JSONL.

Per the 4 distillation design decisions (see docs/archive/ablation_index.zh.md
or finding_a_axis_exploration on "distillation"):
  (1) Target   = cheapest winner            (a)
  (2) Format   = key=value compact text     (iv)
  (3) Input    = bare prompt (default α)    (--k_shot N for β backup)
  (4) No winner = "<UNSALVAGEABLE>" target  (B)

Target example:
    "bl=32 tmpl=answer_marker pos=prefix gen=128 T=0"

Unsalvageable example (prior ceiling-5 or future SS-unresolved prompts):
    "<UNSALVAGEABLE>"

CLI:
    python scripts/validate/ss_to_sft.py --run_dir runs/validation/strategy_search_<ts>
    python scripts/validate/ss_to_sft.py --run_dir <path> --k_shot 3     # few-shot variant
    python scripts/validate/ss_to_sft.py --run_dir <path> --val_frac 0.1

Outputs:
    <run_dir>/sft/sft_train.jsonl
    <run_dir>/sft/sft_val.jsonl
    <run_dir>/sft/sft_stats.json
    <run_dir>/sft/sft_manifest.json    # records decisions + seed for reproducibility
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
SCOPE_FAIL = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
SCOPE_OK = ROOT / "runs" / "validation" / "scope_ok_prompts.json"

UNSALVAGEABLE = "<UNSALVAGEABLE>"
STRATEGY_PREFIX = "\nStrategy: "

# ── Serialization ─────────────────────────────────────────────────────────────

def config_to_keyvalue(cfg: dict) -> str:
    """Fixed key order, compact format. `num_samples` omitted (derivable from T)."""
    T_str = f"{cfg['temperature']:g}"   # "0" not "0.0", "0.3" stays "0.3"
    return (
        f"bl={cfg['block_length']} "
        f"tmpl={cfg['template_name']} "
        f"pos={cfg['template_position']} "
        f"gen={cfg['gen_length']} "
        f"T={T_str}"
    )


def keyvalue_to_config(s: str) -> dict:
    """Inverse of config_to_keyvalue — for round-trip tests / downstream eval."""
    if s.strip() == UNSALVAGEABLE:
        return {}
    parts = dict(kv.split("=", 1) for kv in s.split())
    T = float(parts["T"])
    return {
        "block_length": int(parts["bl"]),
        "template_name": parts["tmpl"],
        "template_position": parts["pos"],
        "gen_length": int(parts["gen"]),
        "temperature": T,
        "num_samples": 1 if T == 0.0 else 4,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prompts_by_group(run_dir: Path) -> dict:
    """Load original scope JSON and index by (group, idx) → prompt text/gt.

    Uses the n used at SS time (stored in run_dir/summary.json → config.n)
    to exactly match the winners.json indices.
    """
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    n = int(summary.get("config", {}).get("n", 60))

    def _load(path: Path, group: str):
        data = json.loads(path.read_text(encoding="utf-8"))[:n]
        return {(group, i): rec for i, rec in enumerate(data)}

    by_idx = {}
    by_idx.update(_load(SCOPE_FAIL, "fail"))
    by_idx.update(_load(SCOPE_OK, "ok"))
    return by_idx


def load_winners(run_dir: Path) -> list:
    p = run_dir / "per_prompt" / "winners.json"
    if not p.exists():
        raise FileNotFoundError(
            f"winners.json missing at {p} — run the final aggregate pass first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


# ── Pair construction ─────────────────────────────────────────────────────────

def build_pair(prompt_text: str, target_str: str) -> dict:
    """Decision (α) — bare prompt + 'Strategy: ' anchor, target = strategy string."""
    return {
        "input": f"Q: {prompt_text}{STRATEGY_PREFIX}",
        "target": target_str,
    }


def build_pair_fewshot(
    prompt_text: str, target_str: str, demos: list[tuple[str, str]],
) -> dict:
    """Decision (β backup) — K demos prepended in-context. Demos exclude abstain.

    `demos` is a list of (prompt_text, target_str) tuples.
    """
    demo_blocks = [
        f"Q: {p}{STRATEGY_PREFIX}{t}"
        for p, t in demos
    ]
    demos_text = "\n\n".join(demo_blocks)
    return {
        "input": f"{demos_text}\n\nQ: {prompt_text}{STRATEGY_PREFIX}",
        "target": target_str,
    }


def _build_all_pairs(
    winners: list, prompts_by_idx: dict,
) -> list[dict]:
    """Collect (group, idx, prompt_text, target_str, is_abstain) records.

    `cheapest` winner used as target (decision a). If no winner exists,
    target = UNSALVAGEABLE (decision B).
    """
    out = []
    for w in winners:
        key = (w["group"], w["idx"])
        rec = prompts_by_idx.get(key)
        if rec is None:
            # shard/aggregate mismatch — skip but flag
            continue
        prompt_text = rec["prompt"]
        cheapest = w.get("winners", {}).get("cheapest")
        if cheapest is None:
            target = UNSALVAGEABLE
            is_abstain = True
        else:
            target = config_to_keyvalue(cheapest["config"])
            is_abstain = False
        out.append({
            "group": w["group"],
            "idx": w["idx"],
            "gt": w.get("gt"),
            "prompt": prompt_text,
            "target": target,
            "is_abstain": is_abstain,
        })
    return out


# ── Splits ────────────────────────────────────────────────────────────────────

def split_train_val(
    records: list[dict], val_frac: float, seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratified split: each group split independently so fail/ok ratio preserved.

    Abstain examples are also split proportionally (B kept, not rerouted).
    """
    rng = random.Random(seed)
    train, val = [], []
    by_group = {"fail": [], "ok": []}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    for group, recs in by_group.items():
        shuffled = list(recs)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_frac))) if shuffled else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


# ── Few-shot demos ────────────────────────────────────────────────────────────

def pick_demos(
    target_rec: dict, demo_pool: list[dict], k: int, seed: int,
) -> list[tuple[str, str]]:
    """Sample k demos from demo_pool, excluding the target itself and abstain
    targets (B decision — don't show abstain in demos).

    Seed derived from (target.group, target.idx) so demos are deterministic
    per target across runs.
    """
    eligible = [
        d for d in demo_pool
        if (d["group"], d["idx"]) != (target_rec["group"], target_rec["idx"])
        and not d["is_abstain"]
    ]
    if len(eligible) < k:
        return []  # not enough; signal caller to degrade to α
    rng = random.Random((seed, target_rec["group"], target_rec["idx"]))
    chosen = rng.sample(eligible, k)
    return [(d["prompt"], d["target"]) for d in chosen]


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(records: list[dict]) -> dict:
    total = len(records)
    by_group = Counter(r["group"] for r in records)
    abstain = sum(1 for r in records if r["is_abstain"])
    strat_counter = Counter(r["target"] for r in records if not r["is_abstain"])
    top5 = strat_counter.most_common(5)
    distinct = len(strat_counter)
    return {
        "total_records": total,
        "by_group": dict(by_group),
        "abstain_count": abstain,
        "abstain_rate": abstain / max(total, 1),
        "distinct_strategies": distinct,
        "top5_strategies": [(s, n, n / max(total - abstain, 1)) for s, n in top5],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="default: <run_dir>/sft/")
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="validation fraction per group (default 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_shot", type=int, default=0,
                    help="decision (β) backup: 0 = bare prompt (α default), "
                         ">0 = K-shot demos")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load winners + prompts
    winners = load_winners(run_dir)
    prompts_by_idx = load_prompts_by_group(run_dir)
    records = _build_all_pairs(winners, prompts_by_idx)
    if not records:
        print("[SS2SFT] no records built; check run_dir / scope files.")
        return

    # Split train / val
    train_recs, val_recs = split_train_val(records, args.val_frac, args.seed)

    # Materialize SFT pairs
    def _materialize(recs: list[dict], demo_pool: list[dict] | None) -> list[dict]:
        out = []
        for r in recs:
            if args.k_shot > 0 and demo_pool is not None:
                demos = pick_demos(r, demo_pool, args.k_shot, args.seed)
                if demos:
                    out.append(build_pair_fewshot(r["prompt"], r["target"], demos))
                    continue  # fall through to α if degraded
            out.append(build_pair(r["prompt"], r["target"]))
        return out

    # Demos for β MUST come from train split (avoid val → train leakage)
    demo_pool = train_recs if args.k_shot > 0 else None
    train_pairs = _materialize(train_recs, demo_pool)
    val_pairs = _materialize(val_recs, demo_pool)

    # Write JSONL
    def _write_jsonl(path: Path, pairs: list[dict]):
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"
    _write_jsonl(train_path, train_pairs)
    _write_jsonl(val_path, val_pairs)

    # Stats + manifest
    stats = {
        "train": compute_stats(train_recs),
        "val": compute_stats(val_recs),
        "overall": compute_stats(records),
    }
    (out_dir / "sft_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "run_dir": str(run_dir),
        "decisions": {
            "target_winner": "cheapest (a)",
            "output_format": "key=value compact (iv)",
            "input_format": "k_shot demos (β)" if args.k_shot > 0 else "bare prompt (α)",
            "no_winner_handling": "<UNSALVAGEABLE> abstain (B)",
        },
        "k_shot": args.k_shot,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "output_anchor": STRATEGY_PREFIX,
        "unsalvageable_token": UNSALVAGEABLE,
        "strategy_format_spec": (
            "bl={int} tmpl={name} pos={prefix|suffix_scaffold|mid_anchor|none} "
            "gen={int} T={float:g}"
        ),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
    }
    (out_dir / "sft_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Report
    print(f"[SS2SFT] run_dir = {run_dir}")
    print(f"[SS2SFT] k_shot  = {args.k_shot}  ({'β few-shot' if args.k_shot else 'α bare'})")
    print(f"[SS2SFT] train   = {len(train_pairs)} pairs → {train_path}")
    print(f"[SS2SFT] val     = {len(val_pairs)} pairs → {val_path}")
    print(f"[SS2SFT] abstain = {stats['overall']['abstain_count']} "
          f"({stats['overall']['abstain_rate']:.2%})")
    print(f"[SS2SFT] distinct strategies (non-abstain): "
          f"{stats['overall']['distinct_strategies']}")
    print(f"[SS2SFT] top-5 strategies:")
    for s, n, frac in stats["overall"]["top5_strategies"]:
        print(f"          {n:>3} ({frac:.2%})  {s}")
    print(f"[SS2SFT] manifest → {out_dir / 'sft_manifest.json'}")
    print(f"[SS2SFT] stats    → {out_dir / 'sft_stats.json'}")


if __name__ == "__main__":
    main()
