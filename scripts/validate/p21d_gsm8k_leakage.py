"""P2.1.d —— GSM8K training-set leakage probe.

动机 (finding_broken_by_answer_is_spurious.zh.md):
  A5 run 里 5/5 broken-by-answer 的 baseline 输出讲的是**完全无关**的 gsm8k 题，
  只是最后数字碰巧撞上 gt。假设 LLaDA-Instruct SFT 里混了 gsm8k train，
  模型在 test prompt 上 fallback 到记忆里的 train 解答。

  LLaDA 论文只声明 gsm8k test 未泄露；train 既没承认也没否认。
  这个脚本直接验证：A5 run 的 tail 输出能否在 gsm8k train 里找到 n-gram 命中。

做法:
  1. Load gsm8k/main train split (7473 条).
  2. 预索引：word-level n-gram (默认 8) → set(train_idx).
  3. 对 A5 per_prompt/*.json 的每条 (idx, template) tail，计算 n-gram 集，
     查倒排表，按命中数排序，取 top-K train 例。
  4. 输出: per-idx / per-template 的 top 命中 + 命中数 + 原 train question。

判定逻辑 (建议):
  - 某条 tail 在 train 中存在 ≥1 个高重叠 (≥5 个共享 8-gram) 的 train 例
    → **确定记忆**，这条 tail 是 SFT 记忆复现。
  - 只有 ≤1 个共享 n-gram → 自然文本巧合。

用法:
  python scripts/validate/p21d_gsm8k_leakage.py \\
      --run_dir runs/validation/a5_prompt_template_20260415_191434 \\
      --out runs/validation/p21d_gsm8k_leakage_191434.json

  # 只扫关键 5 条 broken-by-answer:
  python scripts/validate/p21d_gsm8k_leakage.py \\
      --run_dir runs/validation/a5_prompt_template_20260415_191434 \\
      --only_idx 2,17,22,24,57

依赖:
  pip install datasets  (如果没装; transformers 生态里通常已有)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?|\$[0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenization, keeps numbers and $-prefixed amounts."""
    return _WORD_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def load_gsm8k_train() -> list[dict]:
    """Load gsm8k/main train split. Returns list of {question, answer}."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Need `datasets`: pip install datasets"
        ) from e
    ds = load_dataset("gsm8k", "main", split="train")
    return [{"question": r["question"], "answer": r["answer"]} for r in ds]


def build_index(train: list[dict], n: int) -> dict[tuple[str, ...], list[int]]:
    idx: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        text = r["question"] + "\n" + r["answer"]
        toks = tokenize(text)
        for g in set(ngrams(toks, n)):
            idx[g].append(i)
    return idx


def search_tail(
    tail: str,
    index: dict[tuple[str, ...], list[int]],
    n: int,
    topk: int,
) -> list[dict]:
    toks = tokenize(tail)
    grams = list(set(ngrams(toks, n)))
    if not grams:
        return []
    counter: Counter[int] = Counter()
    for g in grams:
        for train_idx in index.get(g, []):
            counter[train_idx] += 1
    if not counter:
        return []
    top = counter.most_common(topk)
    return [{"train_idx": i, "shared_ngrams": c} for i, c in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="A5 run dir (contains per_prompt/*.json)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. Default: <run_dir>/../p21d_gsm8k_leakage.json",
    )
    ap.add_argument("--ngram", type=int, default=8, help="Word-level n-gram size (default 8)")
    ap.add_argument("--topk", type=int, default=3, help="Top-K train hits per tail")
    ap.add_argument(
        "--only_idx",
        type=str,
        default="",
        help="Comma-separated idx list to limit scan (default: all)",
    )
    ap.add_argument(
        "--min_hits",
        type=int,
        default=3,
        help="Flag a tail as 'likely memorized' if best match shares >= this many n-grams (default 3)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per_prompt_dir = run_dir / "per_prompt"
    if not per_prompt_dir.is_dir():
        raise SystemExit(f"per_prompt dir not found: {per_prompt_dir}")

    out_path = Path(args.out) if args.out else run_dir.parent / f"p21d_gsm8k_leakage_{run_dir.name}.json"

    only_idx: set[int] | None = None
    if args.only_idx.strip():
        only_idx = {int(x) for x in args.only_idx.split(",") if x.strip()}

    print(f"[p21d] loading gsm8k/main train ...")
    train = load_gsm8k_train()
    print(f"[p21d] train size = {len(train)}")

    print(f"[p21d] building {args.ngram}-gram inverted index ...")
    index = build_index(train, args.ngram)
    print(f"[p21d] unique n-grams = {len(index)}")

    pp_files = sorted(per_prompt_dir.glob("*.json"))
    results: dict[str, dict] = {}
    flagged: list[dict] = []

    for fp in pp_files:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        idx = rec["idx"]
        if only_idx is not None and idx not in only_idx:
            continue
        tails = rec.get("tails", {})
        per_template = rec.get("per_template", {})
        entry = {"idx": idx, "gt": rec.get("gt"), "templates": {}}
        for tpl, tail in tails.items():
            hits = search_tail(tail, index, args.ngram, args.topk)
            enriched = []
            best = 0
            for h in hits:
                tr = train[h["train_idx"]]
                enriched.append(
                    {
                        "train_idx": h["train_idx"],
                        "shared_ngrams": h["shared_ngrams"],
                        "question": tr["question"],
                        "answer_tail": tr["answer"][-300:],
                    }
                )
                best = max(best, h["shared_ngrams"])
            entry["templates"][tpl] = {
                "correct": per_template.get(tpl),
                "tail": tail,
                "best_shared_ngrams": best,
                "memorized_flag": best >= args.min_hits,
                "top_hits": enriched,
            }
            if best >= args.min_hits:
                flagged.append(
                    {
                        "idx": idx,
                        "template": tpl,
                        "correct": per_template.get(tpl),
                        "best_shared_ngrams": best,
                        "train_idx": enriched[0]["train_idx"] if enriched else None,
                    }
                )
        results[str(idx)] = entry

    # Summary
    n_tails = sum(len(v["templates"]) for v in results.values())
    n_flagged = len(flagged)

    # Tally: among (correct=True) tails, how many look memorized?
    correct_total = 0
    correct_memorized = 0
    for v in results.values():
        for tpl, t in v["templates"].items():
            if t["correct"]:
                correct_total += 1
                if t["memorized_flag"]:
                    correct_memorized += 1

    summary = {
        "run_dir": str(run_dir),
        "ngram": args.ngram,
        "min_hits": args.min_hits,
        "n_prompts": len(results),
        "n_tails_scanned": n_tails,
        "n_tails_flagged_memorized": n_flagged,
        "correct_tails_total": correct_total,
        "correct_tails_memorized": correct_memorized,
        "correct_memorization_rate": (
            correct_memorized / correct_total if correct_total else 0.0
        ),
    }

    out = {"summary": summary, "flagged": flagged, "per_prompt": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== P2.1.d GSM8K leakage probe ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nflagged tails (best_shared_ngrams >= {args.min_hits}): {n_flagged}")
    for f in flagged[:20]:
        print(f"  idx={f['idx']:>3}  tpl={f['template']:<10} correct={f['correct']}  shared={f['best_shared_ngrams']}  train_idx={f['train_idx']}")
    if n_flagged > 20:
        print(f"  ... and {n_flagged - 20} more")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
