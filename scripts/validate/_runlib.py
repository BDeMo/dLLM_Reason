"""Shared helpers for h1/h2/h3 验证脚本：run_dir 结构、resume、dry_run、增量落盘。

目录结构（每个 validation run 一个 run_dir）：
    <run_dir>/
      config.json        # CLI + model + timestamp
      per_prompt/        # 每条 prompt 一个 json，支持 resume
        0000.json
        ...
      progress.jsonl     # append-only 日志，一行一条
      summary.json       # 跑完后聚合 verdict
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path


class RunDir:
    """统一的 run 目录管理。"""

    def __init__(self, run_dir: Path, hypothesis: str, config: dict,
                 resume: bool = False):
        self.dir = Path(run_dir)
        self.per_prompt = self.dir / "per_prompt"
        self.progress = self.dir / "progress.jsonl"
        self.config_path = self.dir / "config.json"
        self.summary_path = self.dir / "summary.json"
        self.hypothesis = hypothesis
        self.resume = resume

        self.dir.mkdir(parents=True, exist_ok=True)
        self.per_prompt.mkdir(parents=True, exist_ok=True)

        if self.config_path.exists() and resume:
            # 沿用已有 config（避免 resume 时参数对不上）
            existing = json.loads(self.config_path.read_text(encoding="utf-8"))
            print(f"[{hypothesis}] resume from existing config: {self.config_path}")
            self.config = existing
        else:
            self.config = {
                "hypothesis": hypothesis,
                "timestamp_start": datetime.now().isoformat(timespec="seconds"),
                **config,
            }
            self.config_path.write_text(
                json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # ── per-prompt I/O ────────────────────────────────────────────────────────
    def prompt_path(self, idx: int) -> Path:
        return self.per_prompt / f"{idx:04d}.json"

    def has_prompt(self, idx: int) -> bool:
        return self.prompt_path(idx).exists()

    def save_prompt(self, idx: int, record: dict) -> None:
        """保存一条 prompt 的结果（原子写：先 .tmp 再 rename）。"""
        p = self.prompt_path(idx)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        # append progress log
        with self.progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"idx": idx, "ts": datetime.now().isoformat(timespec="seconds"),
                 **{k: v for k, v in record.items() if not isinstance(v, (dict, list))}},
                ensure_ascii=False,
            ) + "\n")

    def load_prompt(self, idx: int) -> dict:
        return json.loads(self.prompt_path(idx).read_text(encoding="utf-8"))

    def load_all_prompts(self) -> list[dict]:
        out = []
        for p in sorted(self.per_prompt.glob("????.json")):
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return out

    # ── summary ───────────────────────────────────────────────────────────────
    def write_summary(self, summary: dict) -> Path:
        summary = {
            **summary,
            "timestamp_end": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(self.dir),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.summary_path


class ProgressPrinter:
    """轻量进度打印（不依赖 tqdm）。"""
    def __init__(self, total: int, tag: str = ""):
        self.total = total
        self.tag = tag
        self.t0 = time.time()
        self.done = 0

    def tick(self, msg: str = "") -> None:
        self.done += 1
        elapsed = time.time() - self.t0
        eta = elapsed / max(self.done, 1) * (self.total - self.done)
        print(f"  [{self.tag}{self.done}/{self.total}] {msg}   "
              f"elapsed={elapsed:.1f}s  eta={eta:.0f}s", flush=True)


def add_common_args(ap) -> None:
    """给 argparse 加通用参数。"""
    ap.add_argument("--run_dir", type=str, default=None,
                    help="结果保存目录；留空则自动 runs/validation/<hypothesis>_<ts>/")
    ap.add_argument("--resume", action="store_true",
                    help="若 run_dir 已有 per_prompt/xxxx.json 则跳过")
    ap.add_argument("--dry_run", action="store_true",
                    help="不加载模型，只打印会跑什么")


def resolve_run_dir(args, hypothesis: str, default_base: Path) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.resume:
        # resume 但没指定 run_dir：取最新
        existing = sorted(default_base.glob(f"{hypothesis}_*"))
        if existing:
            return existing[-1]
    return default_base / f"{hypothesis}_{datetime.now():%Y%m%d_%H%M%S}"
