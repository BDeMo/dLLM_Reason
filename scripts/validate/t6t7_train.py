"""T6/T7 SFT training wrapper — runs Finetuner on a JSONL of (question, answer) pairs.

Thin glue around src/dllm_reason/training/finetune.Finetuner that:
  - Loads LLaDA-8B-Instruct (or a given checkpoint)
  - Loads JSONL produced by t7_gen_correct_samples.py or t6_teacher_trace.py
  - Runs SFT with answer-only loss
  - Saves checkpoints to runs/training/<run_name>/

Usage:
    # T7 SFT on self-distill data
    python scripts/validate/t6t7_train.py \\
        --jsonl_path runs/validation/t7_selfdistill_<ts>/t7_sft.jsonl \\
        --run_name t7_selfdistill \\
        --max_steps 2000 --batch_size 4 --lr 2e-5

    # T6 SFT on teacher traces
    python scripts/validate/t6t7_train.py \\
        --jsonl_path runs/validation/t6_teacher_trace_<ts>/t6_sft.jsonl \\
        --run_name t6_teacher \\
        --max_steps 3000 --batch_size 4 --lr 2e-5

    # 2-stage: T7 → T6 warm-start
    python scripts/validate/t6t7_train.py \\
        --jsonl_path ...t7_sft.jsonl --run_name t7_stage1 --max_steps 1500
    python scripts/validate/t6t7_train.py \\
        --jsonl_path ...t6_sft.jsonl --run_name t6_stage2 \\
        --init_ckpt checkpoints/t7_stage1/best.pt --max_steps 1500

Dry-run to validate data / tokenize without GPU:
    python scripts/validate/t6t7_train.py --jsonl_path ... --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dllm_reason.data.jsonl_dataset import build_jsonl_dataset
from dllm_reason.models.llada import LLaDAWrapper
from dllm_reason.training.finetune import Finetuner, FinetuneConfig
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_path", type=str, required=True,
                    help="JSONL file from t7_gen_correct_samples.py / t6_teacher_trace.py")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--run_name", type=str, default=None,
                    help="output dir name (default: <jsonl_basename>_<ts>)")
    ap.add_argument("--init_ckpt", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                    help="starting model (HF id or local path)")
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                    help="effective batch = batch_size × grad_accum_steps")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--prompt_template", type=str,
                    default="Q: {question}\nA: {answer}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true",
                    help="load data + tokenize + print sizes; skip training")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Resolve run dir ──────────────────────────────────────────────────────
    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{Path(args.jsonl_path).stem}_{ts}"
    save_dir = ROOT / "runs" / "training" / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[T6T7] save_dir = {save_dir}")

    # ── Load model + tokenizer ───────────────────────────────────────────────
    if args.dry_run:
        # avoid loading 8B model for dry-run; stub a HF tokenizer instead
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.init_ckpt,
                                                  trust_remote_code=True)
        model = None
    else:
        print(f"[T6T7] loading model: {args.init_ckpt}")
        model = LLaDAWrapper(model_id=args.init_ckpt,
                             max_seq_len=args.max_seq_len)
        tokenizer = model.tokenizer

    # ── Load dataset ─────────────────────────────────────────────────────────
    train_ds, val_ds = build_jsonl_dataset(
        args.jsonl_path, tokenizer, args.max_seq_len, args.prompt_template,
        train_val_split=args.val_frac, seed=args.seed,
    )
    print(f"[T6T7] train: {len(train_ds)}  val: {len(val_ds) if val_ds else 0}")

    # Sanity: dump first example
    if len(train_ds) > 0:
        ex = train_ds[0]
        n_prompt = int(ex["prompt_mask"].sum())
        n_total = int(ex["attention_mask"].sum())
        print(f"[T6T7] first example: prompt_len={n_prompt}  total_len={n_total}  "
              f"answer_len={n_total - n_prompt}")

    if args.dry_run:
        print("[T6T7] dry-run complete — no training performed")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = None
    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

    # ── Configure + run Finetuner ────────────────────────────────────────────
    cfg = FinetuneConfig(
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=str(save_dir),
        loss_on_answer_only=True,
    )

    # Save config + training meta
    (save_dir / "train_meta.json").write_text(
        json.dumps({
            "cli_args": vars(args),
            "finetune_config": cfg.__dict__,
            "train_size": len(train_ds),
            "val_size": len(val_ds) if val_ds else 0,
            "started_at": datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trainer = Finetuner(model, train_loader, val_loader, cfg)
    trainer.train()

    # ── Export HF-format checkpoint for serving ──────────────────────────────
    # Finetuner saves .pt with {"model_state_dict": ...}, which cannot be loaded
    # by AutoModel.from_pretrained. We additionally save an HF-format dir
    # (config.json + *.safetensors + tokenizer files) so scripts/serve.py and
    # any downstream eval can load the trained ckpt via --model_id.
    hf_dir = save_dir / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    try:
        # LLaDAWrapper exposes the underlying HF model as self.model_internal
        # (or self._model / self.model). Try a few conventional names.
        inner = None
        for attr in ("model_internal", "_model", "model"):
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "save_pretrained"):
                    inner = cand
                    break
        if inner is None:
            raise AttributeError(
                "could not find underlying HF model on LLaDAWrapper "
                "(tried .model_internal, ._model, .model)"
            )
        inner.save_pretrained(hf_dir, safe_serialization=True)
        if hasattr(model, "tokenizer"):
            model.tokenizer.save_pretrained(hf_dir)
        # Copy any trust_remote_code files from the source checkpoint
        # (modeling_llada.py, configuration_llada.py, etc.) if they exist
        src_path = Path(args.init_ckpt)
        if src_path.exists() and src_path.is_dir():
            import shutil
            for name in ("modeling_llada.py", "configuration_llada.py",
                         "tokenization_llada.py"):
                src = src_path / name
                if src.is_file():
                    shutil.copy2(src, hf_dir / name)
        print(f"[T6T7] HF-format ckpt → {hf_dir}")
    except Exception as e:
        print(f"[T6T7] WARN: HF export failed: {e}")
        print(f"[T6T7] .pt checkpoints still usable via load_checkpoint()")

    print(f"[T6T7] done. checkpoints → {save_dir}")


if __name__ == "__main__":
    main()
