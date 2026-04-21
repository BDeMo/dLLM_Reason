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
import os
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
    ap.add_argument("--max_seq_len", type=int, default=768,
                    help="prompt + target length cap. 768 safe for gsm8k "
                         "prompt (100-300 tok) + cleaned teacher trace "
                         "(~100-500 tok). Bump if seeing 'exceeds max_seq_len'.")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True,
                    help="Try enabling HF gradient checkpointing. LLaDA's "
                         "remote code doesn't support it — will fail-soft + "
                         "rely on --use_8bit_adamw for memory savings.")
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--use_8bit_adamw", action="store_true", default=True,
                    help="Replace fp32 AdamW with bitsandbytes AdamW8bit after "
                         "Finetuner init. Saves ~48 GB per rank on 8B model "
                         "(the fp32 moments × 2 that blew budget). Requires "
                         "bitsandbytes installed. Auto-fallback to fp32 if "
                         "bitsandbytes missing.")
    ap.add_argument("--no_8bit_adamw", dest="use_8bit_adamw", action="store_false")
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

    # ── Distributed init (DDP via torchrun) ──────────────────────────────────
    # If launched via `torchrun --nproc_per_node N`, LOCAL_RANK is set per
    # process. Otherwise we run single-GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0
    if is_ddp:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        is_main = (dist.get_rank() == 0)
        print(f"[T6T7] DDP rank={dist.get_rank()}/{world_size}  "
              f"local_rank={local_rank}")
    else:
        world_size = 1
        is_main = True

    def maybe_print(msg):
        if is_main:
            print(msg)

    # ── Resolve run dir ──────────────────────────────────────────────────────
    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{Path(args.jsonl_path).stem}_{ts}"
    save_dir = ROOT / "runs" / "training" / args.run_name
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
    maybe_print(f"[T6T7] save_dir = {save_dir}")

    # ── Load model + tokenizer ───────────────────────────────────────────────
    if args.dry_run:
        # avoid loading 8B model for dry-run; stub a HF tokenizer instead
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.init_ckpt,
                                                  trust_remote_code=True)
        model = None
    else:
        maybe_print(f"[T6T7] loading model: {args.init_ckpt}")
        # In DDP mode: force each rank to load the FULL model on its own
        # single GPU. Vanilla device_map='auto' makes each of the 8 ranks
        # spread the 8B params across all 8 visible GPUs → 8× waste +
        # cross-rank aliasing + OOM. Use device_map={"": local_rank} so
        # the entire model lands on cuda:local_rank.
        #
        # (Setting CUDA_VISIBLE_DEVICES here would be too late — torch.cuda
        # has already initialized inside init_process_group above. The dict
        # device_map is the reliable path.)
        if is_ddp:
            load_kwargs = {"device_map": {"": local_rank}}
            maybe_print(f"[T6T7] DDP: rank {local_rank} loads model on cuda:{local_rank}")
        else:
            load_kwargs = {"device_map": "auto"}
        model = LLaDAWrapper(model_id=args.init_ckpt,
                             max_seq_len=args.max_seq_len,
                             **load_kwargs)
        tokenizer = model.tokenizer

        # Ensure all parameters require grad + model is in train mode.
        model.train()
        n_params = 0
        for p in model.parameters():
            p.requires_grad_(True)
            n_params += p.numel()
        maybe_print(f"[T6T7] model: {n_params/1e6:.1f}M params, all trainable")

        # Try to enable gradient checkpointing on the underlying HF model.
        # LLaDA's remote-code LLaDAModelLM does NOT set
        # _supports_gradient_checkpointing, so gradient_checkpointing_enable()
        # raises ValueError. Make it fail-soft and fall back to 8bit AdamW
        # for the bulk memory savings.
        if args.gradient_checkpointing:
            inner = getattr(model, "_llada", None)
            if inner is not None and hasattr(inner, "gradient_checkpointing_enable"):
                try:
                    inner.gradient_checkpointing_enable()
                    if hasattr(inner, "config"):
                        try: inner.config.use_cache = False
                        except Exception: pass
                    maybe_print(f"[T6T7] gradient_checkpointing_enable() ✓")
                except ValueError as e:
                    maybe_print(f"[T6T7] model doesn't support gradient "
                                f"checkpointing ({e})")
                    maybe_print(f"[T6T7]   will rely on --use_8bit_adamw + bs=1 "
                                f"to fit memory")
            else:
                maybe_print(f"[T6T7] WARN: no ._llada for gradient checkpointing")

        # Wrap in DDP if launched via torchrun
        if is_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP

            # Subclass DDP to forward unknown attribute access to .module.
            # CRITICAL: Finetuner uses self.model.noise_input,
            # self.model.mask_token_id, self.model.device — all custom attrs
            # on LLaDAWrapper. nn.Module.__getattr__ does NOT fall back to
            # .module, so vanilla DDP would AttributeError on every batch.
            class TransparentDDP(DDP):
                def __getattr__(self, name):
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        return getattr(self.module, name)

            # Model already on cuda:local_rank from load_kwargs above;
            # no explicit .to() needed. DDP just wraps.
            model = TransparentDDP(model, device_ids=[local_rank],
                                   find_unused_parameters=False)
            maybe_print(f"[T6T7] wrapped in TransparentDDP on cuda:{local_rank}")

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

    # DataLoader: use DistributedSampler in DDP mode so each rank sees a
    # disjoint shard of the dataset
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        # Finetuner.train() loops on max_steps and re-iters on StopIteration,
        # but doesn't bump sampler.set_epoch — meaning each re-iter shuffles
        # identically. Bump epoch from rank/seed combo so cross-iter order
        # at least varies per rank. (Mild improvement; DDP correctness OK.)
        train_sampler.set_epoch(args.seed)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
    val_loader = None
    if val_ds and len(val_ds) > 0:
        if is_ddp:
            val_sampler = DistributedSampler(val_ds, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    sampler=val_sampler, num_workers=0)
        else:
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

    # Save config + training meta (rank 0 only in DDP mode)
    if is_main:
        (save_dir / "train_meta.json").write_text(
            json.dumps({
                "cli_args": vars(args),
                "finetune_config": cfg.__dict__,
                "train_size": len(train_ds),
                "val_size": len(val_ds) if val_ds else 0,
                "world_size": world_size,
                "started_at": datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    trainer = Finetuner(model, train_loader, val_loader, cfg)

    # ── Swap fp32 AdamW for bitsandbytes AdamW8bit (save ~48 GB / rank) ──────
    # LLaDA can't use gradient checkpointing, so this is the main knob for
    # fitting 8B on A100-80GB. Cosine-with-warm-restarts scheduler is
    # recreated on the new optimizer.
    if args.use_8bit_adamw:
        try:
            import bitsandbytes as bnb
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            trainer.optimizer = bnb.optim.AdamW8bit(
                trainer.model.parameters(), lr=args.lr,
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
            )
            trainer.scheduler = CosineAnnealingWarmRestarts(
                trainer.optimizer, T_0=1000, T_mult=2,
            )
            maybe_print(f"[T6T7] swapped to bitsandbytes AdamW8bit ✓ "
                        f"(saves ~48 GB / rank on 8B model)")
        except ImportError:
            maybe_print(f"[T6T7] WARN: bitsandbytes not installed; "
                        f"sticking with fp32 AdamW. Install: pip install bitsandbytes")

    # ── Resume from latest step_*.pt if present in save_dir ──────────────────
    # Finetuner has load_checkpoint(path) that restores model + optimizer +
    # scheduler + global_step. Pick the highest-step file.
    latest_ckpt = None
    if save_dir.exists():
        step_ckpts = sorted(
            save_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0,
        )
        if step_ckpts:
            latest_ckpt = step_ckpts[-1]
    if latest_ckpt is not None and hasattr(trainer, "load_checkpoint"):
        try:
            trainer.load_checkpoint(latest_ckpt)
            maybe_print(f"[T6T7] resumed from {latest_ckpt.name} "
                        f"(global_step={trainer.global_step})")
        except Exception as e:
            maybe_print(f"[T6T7] WARN: resume from {latest_ckpt} failed: {e!r}")
            maybe_print(f"[T6T7]   starting fresh from init_ckpt")
    elif latest_ckpt is not None:
        maybe_print(f"[T6T7] found {latest_ckpt} but Finetuner lacks "
                    f"load_checkpoint; starting fresh")
    else:
        maybe_print(f"[T6T7] no prior step_*.pt in {save_dir}; fresh start")

    trainer.train()

    # In DDP mode, wait for all ranks to finish training before HF export
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    # ── Export HF-format checkpoint for serving (rank 0 only) ────────────────
    if not is_main:
        return  # non-main ranks exit; rank 0 does the HF export

    hf_dir = save_dir / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    # In DDP, the actual model is wrapped: model.module is LLaDAWrapper.
    inner_wrapper = model.module if is_ddp else model
    try:
        inner = None
        for attr in ("_llada", "_model", "model_internal", "model"):
            if hasattr(inner_wrapper, attr):
                cand = getattr(inner_wrapper, attr)
                if hasattr(cand, "save_pretrained"):
                    inner = cand
                    break
        if inner is None:
            raise AttributeError(
                "could not find underlying HF model on LLaDAWrapper "
                "(tried ._llada, ._model, .model_internal, .model)"
            )
        inner.save_pretrained(hf_dir, safe_serialization=True)
        # Tokenizer lives on inner_wrapper (the LLaDAWrapper); after DDP wrap,
        # the wrapper itself is `model.module`, so use inner_wrapper here.
        if hasattr(inner_wrapper, "tokenizer"):
            inner_wrapper.tokenizer.save_pretrained(hf_dir)
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
