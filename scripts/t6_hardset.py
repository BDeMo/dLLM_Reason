#!/usr/bin/env python
"""Find the true 'hardset' — scope_fail prompts NEVER rescued by any
T6 ckpt (full-SFT × epoch × LoRA rank × epoch) under canonical T=0 pass@1.

Also reports:
  - per-ckpt rescue count + exclusives (rescued only by this ckpt)
  - hardset size + prompts (scope_fail idx, prompt, gt)
  - rescue-count histogram (how many ckpts each fail prompt was rescued by)

Intended use: the hardset is the set that EVEN post-decoding tricks
(pass@N, sampling temperature) will struggle with — they bound the
rescue ceiling under current method. Separate them for deeper analysis
or for verifier-head / tool-use followups.

Reads:
  - runs/validation/t6_ablate/step_*/per_prompt/t6__fail_*.json
  - runs/validation/t6_lora_ablate/r*_step*/per_prompt/t6__fail_*.json
  - runs/validation/scope_fail_prompts.json (for idx → prompt/gt lookup)

Writes:
  - runs/validation/t6_hardset/hardset.json
  - runs/validation/t6_hardset/hardset.md
  - runs/validation/t6_hardset/per_ckpt.json

Usage:
  python scripts/t6_hardset.py
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict


def load_per_prompt(d: Path):
    """Return {fail_idx: correct_bool} for all t6__fail_*.json under d."""
    out = {}
    for f in d.glob("t6__fail_*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            out[int(rec["idx"])] = bool(rec.get("correct", False))
        except Exception:
            pass
    return out


def main():
    root = Path("runs/validation")
    scope_fail_path = root / "scope_fail_prompts.json"
    if not scope_fail_path.is_file():
        print(f"ERROR: {scope_fail_path} not found"); return
    scope_fail = json.loads(scope_fail_path.read_text(encoding="utf-8"))
    N_FAIL = len(scope_fail)

    out_dir = root / "t6_hardset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-ckpt rescue sets
    ckpts = {}   # label -> {idx: correct}
    for pp in sorted((root / "t6_ablate").glob("step_*/per_prompt")):
        label = f"full_{pp.parent.name}"      # full_step_168
        ckpts[label] = load_per_prompt(pp)
    for pp in sorted((root / "t6_lora_ablate").glob("r*_step*/per_prompt")):
        label = f"lora_{pp.parent.name}"       # lora_r1_step672
        ckpts[label] = load_per_prompt(pp)

    if not ckpts:
        print("ERROR: no per_prompt data found"); return

    # Rescue counts: how many ckpts rescued each fail prompt?
    rescue_count = Counter()
    rescuers = defaultdict(list)     # idx -> [ckpt labels that rescued it]
    for label, m in ckpts.items():
        for i, ok in m.items():
            if ok:
                rescue_count[i] += 1
                rescuers[i].append(label)

    # Hardset = fail prompts rescued by ZERO ckpts
    all_idxs = set(range(N_FAIL))
    # Consider only idxs present in at least one per_prompt (guard missing runs)
    covered = set()
    for m in ckpts.values():
        covered.update(m.keys())
    missing = all_idxs - covered
    never_rescued = set(i for i in covered if rescue_count[i] == 0)
    hardset = sorted(never_rescued)

    # Per-ckpt stats
    per_ckpt = {}
    for label, m in ckpts.items():
        rescued = [i for i, ok in m.items() if ok]
        # exclusive = rescued ONLY by this ckpt
        exclusives = [i for i in rescued if rescue_count[i] == 1]
        per_ckpt[label] = {
            "rescued": sorted(rescued),
            "rescued_count": len(rescued),
            "exclusive": sorted(exclusives),
            "exclusive_count": len(exclusives),
        }

    # Rescue-count histogram
    hist = Counter(rescue_count.values())
    hist[0] = len(hardset)  # include never-rescued

    # ── write outputs ────────────────────────────────────────────────────
    # hardset.json: {indices, prompts}
    hardset_detail = [
        {
            "idx": i,
            "prompt": scope_fail[i]["prompt"] if i < len(scope_fail) else None,
            "gt":     (scope_fail[i].get("ground_truth")
                       or scope_fail[i].get("answer")
                       or scope_fail[i].get("gt"))
                      if i < len(scope_fail) else None,
        }
        for i in hardset
    ]
    (out_dir / "hardset.json").write_text(
        json.dumps({
            "n_fail": N_FAIL,
            "n_ckpts": len(ckpts),
            "hardset_size": len(hardset),
            "hardset_indices": hardset,
            "hardset_detail": hardset_detail,
            "never_covered_by_any_run": sorted(missing),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "per_ckpt.json").write_text(
        json.dumps({
            "per_ckpt": per_ckpt,
            "rescue_count_histogram": dict(sorted(hist.items())),
            "rescuers_by_idx": {str(k): v for k, v in sorted(rescuers.items())},
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # hardset.md summary
    lines = [
        "# T6 hardset — fail prompts never rescued by any ckpt",
        "",
        f"- scope_fail size: **{N_FAIL}**",
        f"- ckpts considered: **{len(ckpts)}** "
        f"({sum(1 for k in ckpts if k.startswith('full'))} full-SFT + "
        f"{sum(1 for k in ckpts if k.startswith('lora'))} LoRA)",
        f"- fail prompts rescued by ≥1 ckpt: "
        f"**{N_FAIL - len(hardset) - len(missing)}**",
        f"- **hardset size (rescued by 0)**: **{len(hardset)}** "
        f"({100*len(hardset)/N_FAIL:.1f}% of scope_fail)",
        "",
        "## Rescue-count histogram",
        "",
        "(how many ckpts rescued each fail prompt)",
        "",
        "| # ckpts rescuing | # prompts |",
        "|---|---|",
    ]
    for k in sorted(hist):
        lines.append(f"| {k} | {hist[k]} |")

    lines += [
        "",
        "## Per-ckpt rescue + exclusive contributions",
        "",
        "Exclusive = fail prompts rescued ONLY by this ckpt (not by any other).",
        "",
        "| ckpt | rescued | exclusive |",
        "|---|---|---|",
    ]
    # sort by rescued_count desc
    for label, stats in sorted(per_ckpt.items(),
                                key=lambda x: -x[1]["rescued_count"]):
        lines.append(
            f"| {label} | {stats['rescued_count']}/{N_FAIL} "
            f"| {stats['exclusive_count']} |"
        )

    if hardset:
        lines += [
            "",
            f"## Hardset indices ({len(hardset)} prompts)",
            "",
            f"`{hardset}`",
            "",
            "First 10 prompts (for sanity-check):",
            "",
        ]
        for d in hardset_detail[:10]:
            q = (d.get("prompt") or "")[:180].replace("\n", " ")
            lines.append(f"- **idx {d['idx']}** (gt={d.get('gt')}): {q}…")

    (out_dir / "hardset.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[HARDSET] wrote {out_dir}/hardset.{{json,md}}")


if __name__ == "__main__":
    main()
