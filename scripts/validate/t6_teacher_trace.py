"""T6: AR-teacher → Diffusion-student canvas distill — teacher trace generator.

Queries an AR LM teacher (OpenAI / Anthropic API, or local HuggingFace model)
on gsm8k prompts, asking for solutions in a structured canvas-friendly format:

    <SETUP>briefly list given values and what is asked</SETUP>
    <STEP_1>first calculation step</STEP_1>
    <STEP_2>next calculation step</STEP_2>
    ...
    <ANSWER>numeric answer only, no units</ANSWER>

Filters by <ANSWER> correctness vs ground truth (hard-distill principle —
we only learn from teacher traces whose final answer is right).

Output JSONL (per line):
    {
      "group": "fail", "idx": 4, "gt": "160",
      "question": "<original prompt>",
      "answer": "<full structured trace>",
      "sections": {
        "SETUP":  [start_char, end_char],
        "STEP_1": [start_char, end_char],
        ...
        "ANSWER": [start_char, end_char]
      },
      "teacher_model": "gpt-4o-mini",
      "teacher_answer_correct": true
    }

The 'sections' field is consumed by canvas-aware SFT to optionally weight loss
per region (see src/dllm_reason/data/canvas_sections_dataset.py — TODO).
For first-pass SFT it can be ignored and full 'answer' used as target.

Teacher backends:
  --teacher openai --openai_model gpt-4o-mini
  --teacher anthropic --anthropic_model claude-sonnet-4.5
  --teacher local --local_model Qwen/Qwen2.5-32B-Instruct

Set API key via env: OPENAI_API_KEY / ANTHROPIC_API_KEY.

Usage:
    # generate for FAIL18 with GPT-4o-mini
    python scripts/validate/t6_teacher_trace.py \\
        --groups fail --n 60 --prompt_indices fail18 \\
        --teacher openai --openai_model gpt-4o-mini

    # full fail set + ok set (broader training data)
    python scripts/validate/t6_teacher_trace.py \\
        --groups fail,ok --n 60 --teacher openai --openai_model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from _runlib import RunDir, ProgressPrinter, add_common_args, resolve_run_dir

sys.path.insert(0, str(Path(__file__).parent))
from h1_remask_rescue import is_correct
from strategy_search import _parse_index_spec, FAIL18, CEILING5  # reuse named sets

ROOT = Path(__file__).resolve().parents[2]
SCOPE_FAIL = ROOT / "runs" / "validation" / "scope_fail_prompts.json"
SCOPE_OK = ROOT / "runs" / "validation" / "scope_ok_prompts.json"
OUT_BASE = ROOT / "runs" / "validation"


# ── Teacher prompt + parsing ─────────────────────────────────────────────────

TEACHER_SYSTEM = (
    "You are a careful math tutor. Solve math word problems and present your "
    "solution in a strict structured format so it can be parsed by a script."
)

TEACHER_USER_TEMPLATE = """Solve the following math word problem. Format your solution EXACTLY as:

<SETUP>Briefly list given values and what is asked.</SETUP>
<STEP_1>First calculation step, show arithmetic.</STEP_1>
<STEP_2>Next calculation step.</STEP_2>
... (use as many STEP_N tags as needed, numbered sequentially from 1)
<ANSWER>Numeric answer only. No units. No explanation. Just the number.</ANSWER>

Do not add any text outside these tags. Do not skip tags. If the answer is an
integer, output it without a decimal point.

Problem: {prompt}"""


SECTION_RE = re.compile(r"<(SETUP|STEP_\d+|ANSWER)>(.*?)</\1>", re.DOTALL)


def parse_sections(text: str) -> dict[str, list[int]]:
    """Extract {tag_name: [start_char, end_char]} mappings from a tagged trace.

    Positions are character offsets into the full ``text`` string (so the
    canvas-aware dataset can map them to token positions later).
    """
    out: dict[str, list[int]] = {}
    for m in SECTION_RE.finditer(text):
        tag = m.group(1)
        # inner content positions (between the tags)
        inner_start = m.start(2)
        inner_end = m.end(2)
        out[tag] = [inner_start, inner_end]
    return out


def extract_teacher_answer(text: str) -> str | None:
    """Pull the <ANSWER>...</ANSWER> content."""
    m = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


# ── Teacher backends ─────────────────────────────────────────────────────────

def query_openai(model: str, prompt: str, max_tokens: int = 800,
                 temperature: float = 0.0) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        print("[T6] ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": TEACHER_USER_TEMPLATE.format(prompt=prompt)},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return r.choices[0].message.content or ""


def query_anthropic(model: str, prompt: str, max_tokens: int = 800,
                    temperature: float = 0.0) -> str:
    try:
        import anthropic
    except ImportError:
        print("[T6] ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=TEACHER_SYSTEM,
        messages=[{"role": "user",
                   "content": TEACHER_USER_TEMPLATE.format(prompt=prompt)}],
    )
    # concatenate text blocks
    return "".join(
        b.text for b in r.content if getattr(b, "type", None) == "text"
    )


def query_local(model_id: str, prompt: str, max_tokens: int = 800,
                temperature: float = 0.0,
                _cache: dict = {}) -> str:
    """Lazy-load a local HF model and generate. Cached so we don't reload."""
    if "pipe" not in _cache:
        from transformers import pipeline
        _cache["pipe"] = pipeline(
            "text-generation", model=model_id, torch_dtype="bfloat16",
            device_map="auto",
        )
    pipe = _cache["pipe"]
    full_prompt = (
        f"<|system|>\n{TEACHER_SYSTEM}\n<|user|>\n"
        f"{TEACHER_USER_TEMPLATE.format(prompt=prompt)}\n<|assistant|>\n"
    )
    out = pipe(
        full_prompt,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        temperature=max(temperature, 0.01),
        return_full_text=False,
    )
    return out[0]["generated_text"]


def query_teacher(args, prompt: str) -> str:
    if args.teacher == "openai":
        return query_openai(args.openai_model, prompt, args.max_tokens, args.temperature)
    if args.teacher == "anthropic":
        return query_anthropic(args.anthropic_model, prompt, args.max_tokens,
                               args.temperature)
    if args.teacher == "local":
        return query_local(args.local_model, prompt, args.max_tokens, args.temperature)
    raise ValueError(f"unknown teacher: {args.teacher}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_prompts_for_t6(
    groups: list[str], n: int,
    fail_indices: list[int] | None, ok_indices: list[int] | None,
    scope_path: str | None = None,
    scope_group: str = "gsm8k",
) -> list[tuple[str, int, dict]]:
    """Load prompts.

    If ``scope_path`` is given, read the whole file as a single group labelled
    ``scope_group`` (ignoring fail_indices / ok_indices). This is the 'Full'
    track in the v1.6 plan (gsm8k_train_prompts.json).

    Otherwise, read default fail/ok scope per groups + optional indices.
    """
    out = []
    if scope_path:
        data = json.loads(Path(scope_path).read_text(encoding="utf-8"))
        if n:
            data = data[:n]
        for i, r in enumerate(data):
            out.append((scope_group, i, r))
        return out

    if "fail" in groups:
        all_fails = json.loads(SCOPE_FAIL.read_text(encoding="utf-8"))
        if fail_indices is not None:
            for i in fail_indices:
                out.append(("fail", i, all_fails[i]))
        else:
            for i, r in enumerate(all_fails[:n]):
                out.append(("fail", i, r))
    if "ok" in groups:
        all_oks = json.loads(SCOPE_OK.read_text(encoding="utf-8"))
        if ok_indices is not None:
            for i in ok_indices:
                out.append(("ok", i, all_oks[i]))
        else:
            for i, r in enumerate(all_oks[:n]):
                out.append(("ok", i, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--groups", type=str, default="fail")
    ap.add_argument("--prompt_indices", type=str, default=None,
                    help="fail18 / ceiling5 / explicit 'fail:0,4;ok:2'")
    # Teacher backend
    ap.add_argument("--teacher", type=str, default="openai",
                    choices=["openai", "anthropic", "local"])
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--anthropic_model", type=str, default="claude-sonnet-4.5")
    ap.add_argument("--local_model", type=str,
                    default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries_per_prompt", type=int, default=3,
                    help="retry this many times if answer doesn't match gt "
                         "or format is malformed")
    ap.add_argument("--out_jsonl", type=str, default=None)
    ap.add_argument("--sleep_ms", type=int, default=100,
                    help="sleep between API calls to avoid rate limits")
    ap.add_argument("--scope_path", type=str, default=None,
                    help="override default fail/ok scope; read single scope "
                         "from this JSON (e.g. gsm8k_train_prompts.json)")
    ap.add_argument("--scope_group", type=str, default="gsm8k",
                    help="group label for --scope_path items (default 'gsm8k')")
    add_common_args(ap)
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    fail_idx, ok_idx = _parse_index_spec(args.prompt_indices)
    prompts = load_prompts_for_t6(groups, args.n, fail_idx, ok_idx,
                                   scope_path=args.scope_path,
                                   scope_group=args.scope_group)

    run_dir = resolve_run_dir(args, "t6_teacher_trace", OUT_BASE)
    rd = RunDir(
        run_dir, "T6-TeacherTrace",
        config={
            **vars(args),
            "groups": groups,
            "n_prompts": len(prompts),
        },
        resume=args.resume,
    )
    print(f"[T6] run_dir = {rd.dir}")
    print(f"[T6] prompts: {len(prompts)}  teacher: {args.teacher}/"
          f"{getattr(args, args.teacher + '_model')}")

    out_path = Path(args.out_jsonl) if args.out_jsonl else rd.dir / "t6_sft.jsonl"

    def prompt_key(group: str, i: int) -> str:
        return f"{group}_{i:04d}"

    def is_done(key: str) -> bool:
        return (rd.per_prompt / f"{key}.json").exists()

    todo = [(g, i, r) for (g, i, r) in prompts if not is_done(prompt_key(g, i))]
    print(f"[T6] done={len(prompts) - len(todo)}  todo={len(todo)}")

    if args.dry_run:
        print(f"[T6] DRY RUN — would query teacher on {len(todo)} prompts × "
              f"up to {args.retries_per_prompt} retries each")
        return

    pp = ProgressPrinter(len(todo), tag="T6 ")
    for group, idx, rec in todo:
        prompt, gt = rec["prompt"], rec["ground_truth"]
        key = prompt_key(group, idx)

        attempts = []
        accepted_trace = None
        for attempt in range(args.retries_per_prompt):
            try:
                trace = query_teacher(args, prompt)
            except Exception as e:
                print(f"[T6] WARN: teacher error at {key} attempt {attempt}: {e}")
                time.sleep(1.0)  # back off harder on error
                continue

            ans = extract_teacher_answer(trace)
            ans_ok = (ans is not None) and is_correct(ans, gt)
            sections = parse_sections(trace)
            attempts.append({
                "attempt": attempt,
                "trace": trace,
                "teacher_answer": ans,
                "teacher_answer_correct": ans_ok,
                "sections": sections,
            })
            if ans_ok and ("ANSWER" in sections) and ("SETUP" in sections):
                accepted_trace = attempts[-1]
                break

            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)

        rec_out = {
            "group": group, "idx": idx, "gt": gt,
            "prompt": prompt,
            "teacher": args.teacher,
            "teacher_model": getattr(args, args.teacher + "_model"),
            "accepted": accepted_trace is not None,
            "attempts": attempts,
            "accepted_trace": accepted_trace,
        }
        import os
        path = rd.per_prompt / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec_out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
        status = "✓" if accepted_trace else "✗"
        pp.tick(f"{key} {status}")

    # ── Build SFT JSONL ──────────────────────────────────────────────────────
    all_recs = []
    for group, idx, rec in prompts:
        p = rd.per_prompt / f"{prompt_key(group, idx)}.json"
        if p.exists():
            all_recs.append(json.loads(p.read_text(encoding="utf-8")))

    with out_path.open("w", encoding="utf-8") as f:
        for r in all_recs:
            acc = r.get("accepted_trace")
            if acc is None:
                continue
            sft_pair = {
                "group": r["group"],
                "idx": r["idx"],
                "gt": r["gt"],
                "question": r["prompt"],
                "answer": acc["trace"],
                "sections": acc["sections"],
                "teacher_model": r["teacher_model"],
                "teacher_answer": acc["teacher_answer"],
                "teacher_answer_correct": acc["teacher_answer_correct"],
            }
            f.write(json.dumps(sft_pair, ensure_ascii=False) + "\n")

    n_accepted = sum(1 for r in all_recs if r["accepted"])
    summary = {
        "n_prompts": len(all_recs),
        "n_accepted": n_accepted,
        "cover_rate": n_accepted / max(len(all_recs), 1),
        "sft_pairs_written": n_accepted,
        "out_jsonl": str(out_path),
    }
    rd.write_summary(summary)

    print()
    print("═" * 60)
    print(f"[T6] accepted: {n_accepted}/{len(all_recs)} "
          f"({n_accepted / max(len(all_recs), 1):.2%})")
    print(f"[T6] SFT pairs → {out_path}")
    print(f"[T6] summary   → {rd.summary_path}")


if __name__ == "__main__":
    main()
