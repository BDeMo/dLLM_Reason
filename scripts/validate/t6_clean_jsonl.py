"""T6 teacher trace cleanup — strip Qwen chat-template artifacts.

The raw t6_sft.jsonl produced by t6_teacher_trace.py can contain Qwen's
chat-template tokens (<|system|>, <|end|>, <|start|>, </s>) + duplicate
structured sections in the 'answer' field. This happens because Qwen's
instruct checkpoint emits an internal thinking trace BEFORE the final
structured output when used via HF pipeline.

This script post-processes the JSONL to extract only the clean structured
span (from last <SETUP> through matching </ANSWER>), validating that the
extracted answer still matches ground truth.

Usage:
    python scripts/validate/t6_clean_jsonl.py \\
        --in runs/validation/t6_teacher_trace_<ts>/t6_sft.jsonl

    # default output: adds '_clean' suffix
    → runs/validation/t6_teacher_trace_<ts>/t6_sft_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_remask_rescue import is_correct


# Match a complete <SETUP>...</SETUP> ... <ANSWER>xxx</ANSWER> span.
# Greedy-enough to accept multi-step traces; non-greedy on the inner parts
# to avoid running past a second SETUP/ANSWER pair.
SPAN_RE = re.compile(
    r"<SETUP>.*?</SETUP>(?:\s*<STEP_\d+>.*?</STEP_\d+>)*\s*<ANSWER>(.*?)</ANSWER>",
    re.DOTALL,
)


def extract_clean_span(raw: str, gt: str) -> tuple[str | None, str | None]:
    """Find the best (last-matching, gt-correct) SETUP..ANSWER span.

    Returns (clean_span_text, extracted_answer) or (None, None) if nothing
    valid.
    """
    matches = list(SPAN_RE.finditer(raw))
    if not matches:
        return None, None
    # Prefer the LAST match whose <ANSWER> equals gt; fall back to last match.
    for m in reversed(matches):
        ans = m.group(1).strip()
        if is_correct(ans, gt):
            return m.group(0), ans
    # No gt-matching span; return the last span as best-effort
    last = matches[-1]
    return last.group(0), last.group(1).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="input t6_sft.jsonl from t6_teacher_trace.py")
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default: <in>_clean.jsonl)")
    ap.add_argument("--drop_uncertain", action="store_true",
                    help="drop samples where extracted answer doesn't match "
                         "gt (stricter; default keeps them with a flag).")
    args = ap.parse_args()

    in_path = Path(args.inp)
    if args.out:
        out_path = Path(args.out)
    else:
        # insert '_clean' before '.jsonl'
        out_path = in_path.with_name(in_path.stem + "_clean.jsonl")

    stats = {
        "total": 0,
        "had_span": 0,
        "span_answer_correct": 0,
        "span_answer_wrong_but_kept": 0,
        "no_span_dropped": 0,
        "avg_raw_len": 0,
        "avg_clean_len": 0,
    }
    raw_len_sum = 0
    clean_len_sum = 0

    with in_path.open("r", encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["total"] += 1
            raw_ans = r["answer"] or ""
            gt = str(r.get("gt", ""))
            raw_len_sum += len(raw_ans)

            clean, extracted = extract_clean_span(raw_ans, gt)
            if clean is None:
                stats["no_span_dropped"] += 1
                continue
            stats["had_span"] += 1
            clean_len_sum += len(clean)

            ans_correct = extracted is not None and is_correct(extracted, gt)
            if ans_correct:
                stats["span_answer_correct"] += 1
            else:
                stats["span_answer_wrong_but_kept"] += 1
                if args.drop_uncertain:
                    continue

            out_rec = dict(r)
            out_rec["answer"] = clean
            out_rec["answer_original_len"] = len(raw_ans)
            out_rec["answer_clean_len"] = len(clean)
            out_rec["extracted_answer"] = extracted
            out_rec["extracted_answer_correct"] = ans_correct
            # Sections recompute with offsets relative to the cleaned span
            # (easier downstream than trying to translate raw offsets).
            # Inline a minimal parser to avoid module import gymnastics.
            _sec_re = re.compile(r"<(SETUP|STEP_\d+|ANSWER)>(.*?)</\1>", re.DOTALL)
            sections: dict[str, list[int]] = {}
            for sm in _sec_re.finditer(clean):
                sections[sm.group(1)] = [sm.start(2), sm.end(2)]
            out_rec["sections"] = sections
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    stats["avg_raw_len"] = raw_len_sum / max(stats["total"], 1)
    stats["avg_clean_len"] = clean_len_sum / max(stats["had_span"], 1)
    stats["noise_ratio"] = 1 - (stats["avg_clean_len"] / max(stats["avg_raw_len"], 1))

    print()
    print("═" * 60)
    print(f"[CLEAN] input:       {in_path}")
    print(f"[CLEAN] output:      {out_path}")
    print(f"[CLEAN] total read:  {stats['total']}")
    print(f"[CLEAN] had span:    {stats['had_span']}  "
          f"({stats['had_span']/max(stats['total'],1)*100:.1f}%)")
    print(f"[CLEAN]   correct ans: {stats['span_answer_correct']}")
    print(f"[CLEAN]   wrong ans kept: {stats['span_answer_wrong_but_kept']}")
    if args.drop_uncertain:
        print(f"[CLEAN] drop_uncertain → wrong-ans rows dropped from output")
    print(f"[CLEAN] no-span dropped: {stats['no_span_dropped']}")
    print(f"[CLEAN] avg raw len:   {stats['avg_raw_len']:.0f} chars")
    print(f"[CLEAN] avg clean len: {stats['avg_clean_len']:.0f} chars")
    print(f"[CLEAN] noise removed: {stats['noise_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()
