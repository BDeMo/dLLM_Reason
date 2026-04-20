"""JSONL dataset loader for T6 / T7 SFT training.

Reads JSONL files produced by:
  - scripts/validate/t7_gen_correct_samples.py  (self-distill pairs)
  - scripts/validate/t6_teacher_trace.py         (AR-teacher canvas pairs)

Each line is a JSON object with at least:
  - "question": str — the gsm8k prompt (input)
  - "answer":   str — the target output (correct sample OR structured trace)

Optional keys (T6 only):
  - "sections": dict[str, tuple[int,int]] — section_name → (start_pos, end_pos)
    inside the answer, for canvas-aware loss weighting.

Returned batches are compatible with
src/dllm_reason/training/finetune.py's Finetuner (expects input_ids +
attention_mask + prompt_mask).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from dllm_reason.data.reasoning_datasets import ReasoningDataset
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


def load_jsonl(path: str | Path, filter_fn=None) -> list[dict[str, Any]]:
    """Read a .jsonl file into a list of dicts. Optionally filter by a predicate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if filter_fn is None or filter_fn(rec):
                out.append(rec)
    logger.info(f"Loaded {len(out)} records from {path}")
    return out


class JsonlReasoningDataset(ReasoningDataset):
    """ReasoningDataset backed by a JSONL file. Same tokenization + prompt_mask
    computation as the base class, but data source is a file on disk.

    Extra field on each item: ``meta`` carrying group/idx/gt/temperature/etc.
    for downstream logging and evaluation.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_seq_len: int = 512,
        prompt_template: str = "Q: {question}\nA: {answer}",
        filter_fn=None,
    ):
        data = load_jsonl(jsonl_path, filter_fn=filter_fn)
        # Normalise into ReasoningDataset's expected schema
        norm: list[dict[str, Any]] = []
        for r in data:
            norm.append({
                "question": r["question"],
                "answer": r["answer"],
                # meta is passed through but not used during tokenization
                "_meta": {k: v for k, v in r.items() if k not in ("question", "answer")},
            })
        super().__init__(norm, tokenizer, max_seq_len, prompt_template)
        self.jsonl_path = str(jsonl_path)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        # Attach metadata for logging / eval
        item["_meta"] = self.data[idx].get("_meta", {})
        return item


# ── Convenience factory used by scripts/train.py ─────────────────────────────

def build_jsonl_dataset(
    jsonl_path: str | Path, tokenizer, max_seq_len: int = 512,
    prompt_template: str = "Q: {question}\nA: {answer}",
    train_val_split: float | None = None,
    seed: int = 42,
) -> tuple:
    """Load a JSONL file and optionally split train/val.

    Returns (train_dataset, val_dataset | None).
    """
    all_data = load_jsonl(jsonl_path)
    if train_val_split is None:
        ds = JsonlReasoningDataset(
            jsonl_path, tokenizer, max_seq_len, prompt_template,
        )
        return ds, None

    import random
    rng = random.Random(seed)
    shuffled = list(all_data)
    rng.shuffle(shuffled)

    # Edge case: very few records. Need at least 1 in train to be trainable.
    # If only 1 record exists, put it in train (no val). If ≥ 2, reserve at
    # least 1 for val but always leave at least 1 for train.
    n_total = len(shuffled)
    if n_total < 2:
        val_data = []
        train_data = shuffled
        logger.warning(
            f"Only {n_total} record(s); skipping val split (all in train)."
        )
    else:
        n_val = max(1, int(n_total * train_val_split))
        n_val = min(n_val, n_total - 1)   # leave ≥ 1 for train
        val_data = shuffled[:n_val]
        train_data = shuffled[n_val:]

    # Reuse ReasoningDataset directly with in-memory data (skip re-reading file)
    train_records = [
        {"question": r["question"], "answer": r["answer"]}
        for r in train_data
    ]
    val_records = [
        {"question": r["question"], "answer": r["answer"]}
        for r in val_data
    ]
    train_ds = ReasoningDataset(train_records, tokenizer, max_seq_len, prompt_template)
    val_ds = ReasoningDataset(val_records, tokenizer, max_seq_len, prompt_template)
    logger.info(f"Split: {len(train_ds)} train / {len(val_ds)} val")
    return train_ds, val_ds
