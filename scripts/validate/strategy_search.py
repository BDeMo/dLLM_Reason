"""Strategy Search Phase 1 — per-prompt 策略搜索框架

目标:
  对每条 prompt，在 5 维度 search space 上搜所有 config，存 (prompt, correct_configs)
  pair。Phase 3 可以用这些 pair 做 SFT distill，让模型学会 per-prompt 选策略。

========================================================================
5 个维度（每个含义 + 实现方式 + 默认值域）
========================================================================

E1 已证实 num_steps 加它无收益（rescue=0%），所以 num_steps = gen_length 固定，
不作为独立维度。

1. **block_length** ∈ {16, 32, 64}    (default, 砍了 bl8: A4 里 bl8 broken 多)
   含义: LLaDA block-wise unmasking 块大小
   实现: `block_length` 参数（必须整除 gen_length）

2. **template_name** ∈ {baseline, cot_plain, cot_step, answer_marker, step_by_step_prompt}
   含义: 给模型的指令文本
   值见 TEMPLATES dict

3. **template_position** ∈ {prefix, suffix_scaffold, mid_anchor, none}
   含义: template 在生成流程中的位置（A-axis 从未扫过的新维度）
   实现:
     - "prefix"         : 追加到 prompt（走 /generate）
     - "suffix_scaffold": template 作为 gen region 末尾 anchor（走 /generate_inpaint）
     - "mid_anchor"     : template 作为 gen region 中点 scaffold（走 inpaint）
     - "none"           : 不用 template，纯 base prompt（等价于 baseline × prefix）
   去重: position=none 时 template_name 强制为 baseline（避免冗余）

4. **gen_length** ∈ {128, 160, 192}   (default, 砍了 g64/g96/g256: A6 里弱档位)
   含义: 生成 token 数 = diffusion 步数（coupled，E1 证实独立加步数无效）

5. **temperature** ∈ {0.0, 0.3, 0.7}  (default, 砍了 T=1.0: H3 里 T=0.7 已够)
   实现: Gumbel noise scale
   **pass@N 统一策略**: T=0 → num_samples=1 (deterministic 不重复)
                       T>0 → num_samples=4 (pass@1/pass@4 聚合)

========================================================================
Budget 估算（默认空间）
========================================================================
合法 (block_length, gen_length) pair（gen % bl == 0）:
  bl=16: gen∈{128,160,192} → 3
  bl=32: gen∈{128,160,192} → 3
  bl=64: gen∈{128,192}     → 2
  → 8 (bl, gen) pairs

合法 (template_name, template_position) 组合（position=none 强制 baseline）:
  baseline × {prefix, suffix_scaffold, mid_anchor, none} → 4
  {cot_plain, cot_step, answer, step_prompt} × {prefix, suffix_scaffold, mid_anchor} → 12
  → 16 组合

Total configs = 8 × 16 × 3 = 384 configs/prompt
Total calls:
  T=0 configs: 8 × 16 = 128 × N=1 = 128
  T>0 configs: 8 × 16 × 2 = 256 × N=4 = 1024
  per prompt = 1152 calls
  × 109 prompts = ~125k calls
  @1.5s/call = ~52h ≈ 2.2 天
  @1.0s/call = ~35h ≈ 1.5 天

用 --full_space 切到完整 {8,16,32,64} × {64,96,128,160,192,256} × {0,0.3,0.7,1.0} → ~5 天

========================================================================
Winners（多种口径分别保存）—— 每条 prompt 保存 4 种 winner + all_correct list
========================================================================

winners.json 每条 prompt 一条记录：
  {
    "group": "fail"/"ok", "idx": 0, "gt": "70000",
    "n_correct_configs": 7,
    "all_correct_config_ids": [...],          # 所有能答对的 config_id 列表
    "winners": {
      "cheapest":       {..config..},          # 最低 compute cost (bl × gen × N)
      "shortest":       {..config..},          # 输出 tail 最短的 correct config
      "most_reliable":  {..config..},          # pass@1 最高 (T>0 稳定性)
      "deterministic":  {..config..},          # T=0 下能对的最低 cost config
    }
  }

4 种 winner 覆盖 4 种 distillation use case：
  - cheapest:      inference 最便宜部署
  - shortest:      倾向 concise 答案
  - most_reliable: T>0 sampling 最稳
  - deterministic: T=0 一把过

========================================================================
CLI
========================================================================

  # 默认空间（推荐）dry-run 预估 cost
  python scripts/validate/strategy_search.py --n 5 --dry_run

  # 跑全 scope（109=60 fail + 49 ok）
  python scripts/validate/strategy_search.py --n 60 --groups fail,ok

  # 切到完整值域（慢）
  python scripts/validate/strategy_search.py --n 60 --groups fail,ok --full_space

  # 断点 resume
  python scripts/validate/strategy_search.py --run_dir <existing> --resume

  # 覆盖某一维（例如固定 bl=32 做 ablation）
  python scripts/validate/strategy_search.py --n 60 --values block_length=32
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir
from _http_client import ValidationAPIClient, add_server_arg

sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import is_correct

ROOT = Path(__file__).resolve().parents[2]
SCOPE_FAIL = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
SCOPE_OK = ROOT / "runs" / "validation" / "scope_ok_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"

# ── Search space 定义 (default values) ────────────────────────────────────────

TEMPLATES: dict[str, str] = {
    # name          → text appended after user question (for prefix mode)
    "baseline": "",  # pure question, no template
    "cot_plain": "\n\nLet's think step by step.",
    "cot_step": "\n\nStep 1: Let me break this down.\nStep 2: Now I'll compute.\nStep 3: Then verify.",
    "answer_marker": "\n\nAnswer:",
    "step_by_step_prompt": "\n\nLet's solve this step by step, showing our work:",
}

POSITIONS = ["prefix", "suffix_scaffold", "mid_anchor", "none"]

# ── Default search space (pruned per A-axis findings) ─────────────────────────
# Prunings vs full grid:
#   block_length — dropped bl=8 (A4 里 bl8 broken 多, rescue 只贡献 idx=53 一条)
#   gen_length   — dropped g64/g96/g256 (A6 里这三档都弱于 g128 baseline)
#   temperature  — dropped T=1.0 (H3 里 T=0.7 已够 diversity)
#   num_steps    — E1 证实独立加步数零贡献，固定 = gen_length
DEFAULT_SPACE = {
    "block_length": [16, 32, 64],
    "template_name": list(TEMPLATES.keys()),
    "template_position": POSITIONS,
    "gen_length": [128, 160, 192],
    "temperature": [0.0, 0.3, 0.7],
}

# Full grid — 启用 --full_space 时切到这个
FULL_SPACE = {
    "block_length": [8, 16, 32, 64],
    "template_name": list(TEMPLATES.keys()),
    "template_position": POSITIONS,
    "gen_length": [64, 96, 128, 160, 192, 256],
    "temperature": [0.0, 0.3, 0.7, 1.0],
}

DIMS = list(DEFAULT_SPACE.keys())


# ── Config dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    block_length: int
    template_name: str
    template_position: str
    gen_length: int
    temperature: float
    num_samples: int  # pass@N: 1 for T=0, >=4 for T>0

    @property
    def id(self) -> str:
        return (
            f"bl{self.block_length}"
            f"_tmpl-{self.template_name}"
            f"_pos-{self.template_position}"
            f"_gen{self.gen_length}"
            f"_T{self.temperature:g}"
            f"_n{self.num_samples}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ── Config enumeration with validity constraints ─────────────────────────────

def _num_samples_for_temp(T: float, fixed_n: int | None = None) -> int:
    """T=0 → 1 sample (deterministic); T>0 → pass@N (4 default)."""
    if fixed_n is not None:
        return fixed_n
    return 1 if T == 0.0 else 4


def enumerate_configs(space: dict, fixed_n_samples: int | None = None) -> list[StrategyConfig]:
    """Enumerate all valid StrategyConfig combinations within ``space``.

    Constraints applied (enforced here — do NOT let downstream hit errors):
      - gen_length must be divisible by block_length
      - template_position="none" is only emitted once per (bl, gen, T) combo
        (regardless of template_name, since none means "no template")
      - num_samples picked per temperature (deterministic vs pass@N)
    """
    configs: list[StrategyConfig] = []
    seen: set[str] = set()

    for bl, tname, tpos, gen, T in itertools.product(
        space["block_length"],
        space["template_name"],
        space["template_position"],
        space["gen_length"],
        space["temperature"],
    ):
        # Constraint: gen_length divisible by block_length
        if gen % bl != 0:
            continue

        # Normalise template_name for position="none":
        # it doesn't matter which template we picked, the result is the same.
        effective_tname = tname if tpos != "none" else "baseline"

        n = _num_samples_for_temp(T, fixed_n_samples)
        cfg = StrategyConfig(
            block_length=bl,
            template_name=effective_tname,
            template_position=tpos,
            gen_length=gen,
            temperature=T,
            num_samples=n,
        )
        if cfg.id in seen:
            continue
        seen.add(cfg.id)
        configs.append(cfg)
    return configs


# ── Call dispatch: each template_position maps to a different endpoint ───────

def build_prefix_prompt(base_prompt: str, template_text: str) -> str:
    """template_position='prefix': 追加 template 到 user question 末尾。"""
    if not template_text:
        return base_prompt
    return base_prompt + template_text


def build_inpaint_anchors(
    template_text: str, position: str, gen_length: int,
    approx_anchor_tokens: int = 12,
) -> list[tuple[int, str]]:
    """Map template_position={suffix_scaffold, mid_anchor} to anchor list.

    ``approx_anchor_tokens`` is used to reserve space at the tail for
    ``suffix_scaffold`` — we can't know the exact token count without
    tokenizing, so we pick a conservative constant and let the server
    truncate if needed.
    """
    if not template_text or position == "none":
        return []
    if position == "suffix_scaffold":
        # Place anchor near the end, leaving room so it fits inside gen_length
        start = max(0, gen_length - approx_anchor_tokens)
        return [(start, template_text)]
    if position == "mid_anchor":
        return [(gen_length // 2, template_text)]
    raise ValueError(f"unknown inpaint position: {position}")


def run_one_config(
    api: ValidationAPIClient,
    prompt: str,
    cfg: StrategyConfig,
) -> dict:
    """Execute a single StrategyConfig. Returns dict with correct_list, pass@k,
    and truncated tails. The 5 dimensions map onto concrete endpoint calls:
      prefix / none → /generate
      suffix_scaffold / mid_anchor → /generate_inpaint
    """
    template_text = TEMPLATES[cfg.template_name]

    # Route to endpoint based on template_position
    correct_list: list[bool] = []
    tails: list[str] = []
    for _ in range(cfg.num_samples):
        if cfg.template_position in ("prefix", "none"):
            # template merged into prompt prefix (or omitted entirely)
            effective_prompt = build_prefix_prompt(
                prompt, template_text if cfg.template_position == "prefix" else ""
            )
            out = api.generate(
                effective_prompt, strategy="confidence",
                max_new_tokens=cfg.gen_length,
                num_steps=cfg.gen_length,  # coupled as in A6
                block_length=cfg.block_length,
                temperature=cfg.temperature,
            )
        else:
            # suffix_scaffold / mid_anchor → inpainting endpoint
            anchors = build_inpaint_anchors(
                template_text, cfg.template_position, cfg.gen_length
            )
            out = api.generate_inpaint(
                prompt,
                anchors=anchors,
                gen_length=cfg.gen_length,
                steps=cfg.gen_length,
                block_length=cfg.block_length,
                temperature=cfg.temperature,
            )
        tails.append(out[-200:])
        correct_list.append(False)  # set below via is_correct
    return {"correct_list": correct_list, "tails": tails}


def pass_at_k(corrects: list[bool], k: int) -> float:
    return 1.0 if any(corrects[:k]) else 0.0


# ── Winner picking: 4 different "best config" definitions ────────────────────

def _compute_cost(config: dict) -> float:
    """Proxy for inference cost: block_length × gen_length × num_samples.
    Lower bl → more blocks → more forward passes, so we use gen_length/bl
    as a block-count proxy. Actual LLaDA cost ≈ gen_length × num_samples
    (blocks only affect schedule, not total steps since num_steps=gen_length),
    so the main cost drivers are gen_length and N.
    """
    return config["gen_length"] * config["num_samples"]


def _pick_winners(prompt_rec: dict) -> dict:
    """Pick 4 winner configs per prompt (分别保存):
      - cheapest:      min compute cost (gen_length × num_samples) s.t. pass@1
      - shortest:      min output tail length s.t. pass@1
      - most_reliable: max fraction of correct samples within N (robust to T>0)
      - deterministic: min cost among T=0 correct configs (deployable greedy)

    Returns a dict with group/idx/gt + "all_correct_config_ids" list + "winners"
    dict keyed by the 4 kinds. If no config is correct, winners={} (empty).
    """
    passed = [res for res in prompt_rec["results"] if res.get("pass@1", 0) >= 1.0]
    all_correct_ids = [r["config_id"] for r in passed]

    def _mean_correct(r: dict) -> float:
        cl = r.get("correct_list", [])
        return sum(1 for c in cl if c) / max(len(cl), 1)

    def _tail_len(r: dict) -> int:
        # pick the correct sample's tail (first correct one); len = character count
        cl = r.get("correct_list", [])
        for i, c in enumerate(cl):
            if c and i < len(r.get("tails", [])):
                return len(r["tails"][i])
        return 10**9  # fallback huge, shouldn't happen since r is in passed

    winners: dict = {}
    if passed:
        winners["cheapest"] = min(passed, key=lambda r: _compute_cost(r["config"]))
        winners["shortest"] = min(passed, key=_tail_len)
        # most_reliable: highest fraction correct, tiebreak by lowest cost
        winners["most_reliable"] = max(
            passed,
            key=lambda r: (_mean_correct(r), -_compute_cost(r["config"])),
        )
        t0_correct = [r for r in passed if r["config"]["temperature"] == 0.0]
        if t0_correct:
            winners["deterministic"] = min(
                t0_correct, key=lambda r: _compute_cost(r["config"])
            )

    # Strip verbose fields to keep winners.json lightweight
    def _slim(r: dict) -> dict:
        return {
            "config_id": r["config_id"],
            "config": r["config"],
            "pass@1": r.get("pass@1"),
            "pass@4": r.get("pass@4"),
        }

    return {
        "group": prompt_rec["group"],
        "idx": prompt_rec["idx"],
        "gt": prompt_rec["gt"],
        "n_correct_configs": len(passed),
        "all_correct_config_ids": all_correct_ids,
        "winners": {k: _slim(v) for k, v in winners.items()},
    }


# ── CLI value parsing ────────────────────────────────────────────────────────

def parse_values_overrides(values_list: list[str], space: dict) -> dict:
    """Parse --values dim=v1,v2,... overrides into the space dict."""
    out = {k: list(v) for k, v in space.items()}
    for item in values_list:
        if "=" not in item:
            raise ValueError(f"--values expects 'dim=v1,v2,...', got {item!r}")
        dim, vs = item.split("=", 1)
        dim = dim.strip()
        if dim not in space:
            raise ValueError(f"unknown dim {dim!r} (valid: {list(space)})")
        raw_vals = [v.strip() for v in vs.split(",") if v.strip()]
        # cast to the same type as the first default value
        sample = space[dim][0]
        if isinstance(sample, int):
            parsed = [int(v) for v in raw_vals]
        elif isinstance(sample, float):
            parsed = [float(v) for v in raw_vals]
        else:
            parsed = raw_vals
        out[dim] = parsed
    return out


def parse_dims_filter(dims_arg: str) -> list[str]:
    """--dims b,c,d restricts which dimensions VARY. Fixed dims get a single
    default value (their first entry in DEFAULT_SPACE). This is different from
    --values: --dims controls WHAT IS SEARCHED, --values fills in the list."""
    if not dims_arg:
        return DIMS
    asked = [d.strip() for d in dims_arg.split(",") if d.strip()]
    for d in asked:
        if d not in DIMS:
            raise ValueError(f"unknown dim {d!r} in --dims (valid: {DIMS})")
    return asked


# ── Main ─────────────────────────────────────────────────────────────────────

def load_prompts(groups: list[str], n: int) -> list[tuple[int, str, dict]]:
    """Return (group, original_idx, rec) tuples. Group is 'fail' or 'ok'."""
    out = []
    if "fail" in groups:
        fails = json.loads(SCOPE_FAIL.read_text(encoding="utf-8"))[:n]
        for i, r in enumerate(fails):
            out.append(("fail", i, r))
    if "ok" in groups:
        oks = json.loads(SCOPE_OK.read_text(encoding="utf-8"))[:n]
        for i, r in enumerate(oks):
            out.append(("ok", i, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                    help="top-N from each group (fail/ok). Use 60 for full fail set, "
                         "49 for full ok set (scope has 60 fail + 49 ok)")
    ap.add_argument("--groups", type=str, default="fail,ok",
                    help="comma-separated: 'fail', 'ok', or 'fail,ok' (default: both)")
    ap.add_argument("--dims", type=str, default=",".join(DIMS),
                    help=f"which dims to VARY (default all). Options: {DIMS}")
    ap.add_argument("--values", action="append", default=[],
                    help="override per-dim values, e.g. 'block_length=32' "
                         "(repeat --values for each dim)")
    ap.add_argument("--fixed_n_samples", type=int, default=None,
                    help="override auto pick; if set, all configs use this num_samples")
    ap.add_argument("--full_space", action="store_true",
                    help="use full grid {bl=8,16,32,64}×{gen=64..256}×{T=0,0.3,0.7,1.0} "
                         "(ignores pruned defaults; ~5 days at 1.5s/call)")
    # Multi-GPU sharding (Scheme A: shared run_dir, slice prompts per GPU) ──────
    ap.add_argument("--prompt_start", type=int, default=None,
                    help="shard slice start (inclusive) into the loaded prompts "
                         "list after --groups/--n. Use with --prompt_end to run a "
                         "subset of prompts per GPU. Sharding is SAFE: each prompt "
                         "is scored independently, per_prompt/{group}_{idx}.json "
                         "files don't collide.")
    ap.add_argument("--prompt_end", type=int, default=None,
                    help="shard slice end (exclusive); defaults to len(prompts).")
    ap.add_argument("--skip_summary", action="store_true",
                    help="skip winners.json + summary.json aggregation at end. "
                         "REQUIRED for shard workers to avoid race-writing the "
                         "shared summary files. Run a final no-slice --resume "
                         "pass without this flag to build the global summary.")
    add_common_args(ap)
    add_server_arg(ap)
    args = ap.parse_args()

    # Build search space: start from defaults (or full), narrow via --dims / --values
    base_space = FULL_SPACE if args.full_space else DEFAULT_SPACE
    dims_to_vary = parse_dims_filter(args.dims)
    space = {k: list(v) for k, v in base_space.items()}
    # Freeze non-varying dims to their first default
    for d in DIMS:
        if d not in dims_to_vary:
            space[d] = [space[d][0]]
    space = parse_values_overrides(args.values, space)

    configs = enumerate_configs(space, fixed_n_samples=args.fixed_n_samples)
    if not configs:
        print("[SS] empty search space after constraints; abort")
        return

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    prompts_full = load_prompts(groups, args.n)
    if not prompts_full:
        print("[SS] no prompts loaded; check --groups / --n")
        return

    # Apply shard slice to the WORK list (todo). Summary at the end always
    # aggregates from the FULL list so shard workers + final pass agree.
    if args.prompt_start is not None or args.prompt_end is not None:
        s = args.prompt_start if args.prompt_start is not None else 0
        e = args.prompt_end if args.prompt_end is not None else len(prompts_full)
        prompts = prompts_full[s:e]
        print(f"[SS] shard slice: prompts[{s}:{e}] → {len(prompts)} prompts "
              f"(of {len(prompts_full)} total)")
    else:
        prompts = prompts_full

    total_calls = sum(c.num_samples for c in configs) * len(prompts)
    est_hours_15 = total_calls * 1.5 / 3600
    est_hours_10 = total_calls * 1.0 / 3600
    print(f"[SS] groups={groups}  prompts={len(prompts)}  configs/prompt={len(configs)}")
    print(f"[SS] total HTTP calls (best case, no resume): {total_calls:,}")
    print(f"[SS] wall-time estimate: ~{est_hours_10:.1f}h @1.0s/call, "
          f"~{est_hours_15:.1f}h @1.5s/call")
    if est_hours_15 > 48:
        print(f"[SS] ⚠  budget > 48h. Consider --values to narrow dims or drop --full_space.")
    print(f"[SS] dims varying: {dims_to_vary}  "
          f"({'FULL_SPACE' if args.full_space else 'DEFAULT_SPACE (pruned)'})")
    print(f"[SS] space:")
    for d in DIMS:
        vals = space[d]
        print(f"       {d}: {vals}  ({'var' if d in dims_to_vary else 'fixed'})")

    run_dir = resolve_run_dir(args, "strategy_search", OUT_BASE)
    rd = RunDir(
        run_dir, "StrategySearch",
        config={
            **vars(args),
            "dims_varying": dims_to_vary,
            "space": space,
            "configs_n": len(configs),
            "total_samples": sum(c.num_samples for c in configs),
        },
        resume=args.resume,
    )
    print(f"[SS] run_dir = {rd.dir}")

    # Each per_prompt file keyed by flat index across groups
    def prompt_key(group: str, i: int) -> str:
        return f"{group}_{i:04d}"

    # Determine which prompts are fully done (= all configs cached)
    def is_done(key: str) -> bool:
        p = rd.per_prompt / f"{key}.json"
        if not p.exists():
            return False
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            done_ids = {r["config_id"] for r in rec.get("results", [])}
            return all(cfg.id in done_ids for cfg in configs)
        except Exception:
            return False

    todo = [(g, i, r) for (g, i, r) in prompts if not is_done(prompt_key(g, i))]
    print(f"[SS] prompts done={len(prompts) - len(todo)}  todo={len(todo)}")

    if args.dry_run:
        print(f"[SS] DRY RUN — would run {sum(c.num_samples for c in configs) * len(todo)} "
              f"HTTP calls across {len(todo)} prompts × {len(configs)} configs")
        # show first 5 configs for sanity
        print("[SS] first 5 configs:")
        for c in configs[:5]:
            print(f"    {c.id}")
        return

    api = ValidationAPIClient(args.server_url)
    api.check_health()

    pp = ProgressPrinter(len(todo), tag="SS ")
    for group, i, rec in todo:
        prompt, gt = rec["prompt"], rec["ground_truth"]
        key = prompt_key(group, i)

        # Preserve previously-cached results (resume)
        existing = {}
        path = rd.per_prompt / f"{key}.json"
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                existing = {r["config_id"]: r for r in old.get("results", [])}
            except Exception:
                existing = {}

        results = []
        n_correct_cfg = 0
        for cfg in configs:
            if cfg.id in existing:
                results.append(existing[cfg.id])
                if existing[cfg.id].get("pass@1", 0) >= 1.0:
                    n_correct_cfg += 1
                continue
            raw = run_one_config(api, prompt, cfg)
            # score correctness per sample
            for s_i, tail in enumerate(raw["tails"]):
                raw["correct_list"][s_i] = bool(is_correct(tail, gt))
            p1 = pass_at_k(raw["correct_list"], 1)
            p4 = pass_at_k(raw["correct_list"], min(4, cfg.num_samples))
            p8 = pass_at_k(raw["correct_list"], min(8, cfg.num_samples))
            if p1 >= 1.0:
                n_correct_cfg += 1
            results.append({
                "config_id": cfg.id,
                "config": cfg.to_dict(),
                "correct_list": raw["correct_list"],
                "pass@1": p1,
                "pass@4": p4,
                "pass@8": p8,
                "tails": raw["tails"],
            })

        rec_out = {
            "idx": i,
            "group": group,
            "gt": gt,
            "n_configs": len(configs),
            "n_configs_passed": n_correct_cfg,
            "results": results,
        }
        # Atomic write
        import os
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec_out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
        pp.tick(f"{key}  {n_correct_cfg}/{len(configs)} configs passed")

    # Summary aggregation — use FULL prompts list so shard workers + final
    # aggregate pass agree on scope. Shard workers should pass --skip_summary
    # to avoid racing on winners.json / summary.json writes.
    if args.skip_summary:
        print()
        print(f"[SS] --skip_summary set; shard done, skipping aggregate. "
              f"Run a final `--resume` pass without --prompt_start/--prompt_end "
              f"and without --skip_summary to build global summary.")
        return

    all_recs = []
    missing = 0
    for group, i, rec in prompts_full:
        p = rd.per_prompt / f"{prompt_key(group, i)}.json"
        if p.exists():
            all_recs.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            missing += 1
    if missing:
        print(f"[SS] WARN: {missing}/{len(prompts_full)} per_prompt files missing; "
              f"summary will be partial.")

    oracle_correct = sum(1 for r in all_recs if r["n_configs_passed"] > 0)
    winners = [_pick_winners(r) for r in all_recs]
    n_distillable = sum(1 for w in winners if w["winners"])

    (rd.per_prompt / "winners.json").write_text(
        json.dumps(winners, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Per-category winner stats (for logging + downstream picking)
    winner_kinds = ("cheapest", "shortest", "most_reliable", "deterministic")
    kind_counts = {k: sum(1 for w in winners if k in w["winners"]) for k in winner_kinds}

    summary = {
        "n_prompts": len(all_recs),
        "n_configs_per_prompt": len(configs),
        "oracle_correct": oracle_correct,
        "oracle_rate": oracle_correct / max(len(all_recs), 1),
        "n_distillable": n_distillable,
        "winner_counts_by_kind": kind_counts,
        "space": space,
        "config": rd.config,
    }
    rd.write_summary(summary)

    print()
    print("═" * 60)
    print(f"[SS] Oracle correct: {oracle_correct}/{len(all_recs)} "
          f"({oracle_correct / max(len(all_recs), 1):.2%})")
    print(f"[SS] Distillable (≥1 correct config): {n_distillable}")
    print(f"[SS] winner kind counts (prompts with non-empty pick):")
    for k in winner_kinds:
        print(f"       {k:<15}: {kind_counts[k]}")
    print(f"[SS] winners → {rd.per_prompt / 'winners.json'}")
    print(f"[SS] summary → {rd.summary_path}")


if __name__ == "__main__":
    main()
