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
                         "remote code doesn't support it — fail-soft.")
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--parallel", type=str, default="fsdp",
                    choices=["ddp", "fsdp"],
                    help="fsdp (default, recommended for 8B on 80GB A100): "
                         "FULL_SHARD splits weights/grads/optim-state across "
                         "ranks → ~30 GB/rank for 8B. ddp: each rank holds "
                         "full model+grads+optim → OOM at 8B/80GB unless "
                         "--use_lora.")
    ap.add_argument("--use_lora", action="store_true",
                    help="Apply LoRA adapters (peft). Freezes base weights; "
                         "only LoRA params trainable → optim-state shrinks "
                         "~1000×. Recommended for single-GPU or ddp on 80GB.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target", type=str,
                    default="q_proj,k_proj,v_proj,o_proj",
                    help="comma-separated linear module names to wrap")
    ap.add_argument("--lora_merge_on_save", action="store_true", default=True,
                    help="merge LoRA into base weights before HF export "
                         "(so downstream serve.py loads merged checkpoint "
                         "without peft dep)")
    ap.add_argument("--no_lora_merge_on_save", dest="lora_merge_on_save",
                    action="store_false")
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                    help="effective batch = batch_size × grad_accum_steps")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--keep_last_n", type=int, default=2,
                    help="rolling: keep only the most recent N step_*.pt "
                         "(best.pt always kept). 8B+fp32 AdamW ckpt ~96GB; "
                         "default 2 caps disk at ~200GB + best.pt. 0=keep all")
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
        # bf16 load: avoids the 32 GB fp32 → bf16 peak before FSDP shards.
        # FSDP MixedPrecision recasts on the fly anyway; DDP keeps bf16.
        if is_ddp:
            load_kwargs = {"device_map": {"": local_rank},
                           "torch_dtype": torch.bfloat16}
            maybe_print(f"[T6T7] {('FSDP' if args.parallel=='fsdp' else 'DDP')}: "
                        f"rank {local_rank} loads bf16 on cuda:{local_rank}")
        else:
            load_kwargs = {"device_map": "auto",
                           "torch_dtype": torch.bfloat16}
        model = LLaDAWrapper(model_id=args.init_ckpt,
                             max_seq_len=args.max_seq_len,
                             **load_kwargs)
        tokenizer = model.tokenizer

        model.train()

        # ── LoRA (optional, applied BEFORE FSDP/DDP wrap) ────────────────────
        # peft.get_peft_model replaces target nn.Linear with LoraLinear,
        # freezes base weights, and marks only lora_A/lora_B trainable.
        # With --use_lora, optimizer state for the base 8B model is ZERO
        # (base params have requires_grad=False); only ~10-50M LoRA params
        # carry AdamW moments → trivially fits 80GB even without FSDP.
        if args.use_lora:
            from peft import LoraConfig, get_peft_model
            targets = [s.strip() for s in args.lora_target.split(",") if s.strip()]
            lcfg = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha,
                target_modules=targets, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM",
            )
            try:
                model._llada = get_peft_model(model._llada, lcfg)
            except ValueError as e:
                # peft raises "Target modules {...} not found in the base
                # model" if --lora_target doesn't match LLaDA's actual
                # linear names. LLaDA may use OLMo-style fused `att_proj`
                # / `ff_proj` instead of Llama's `q_proj,k_proj,...`.
                # Give a helpful error with the actual module names.
                actual = sorted({n.split(".")[-1]
                                 for n, m in model._llada.named_modules()
                                 if isinstance(m, torch.nn.Linear)})
                raise ValueError(
                    f"LoRA target_modules {targets} not found in LLaDA.\n"
                    f"Linear module names present: {actual}\n"
                    f"Retry with --lora_target '<name1>,<name2>,...'"
                ) from e
            n_trainable = sum(p.numel() for p in model._llada.parameters()
                              if p.requires_grad)
            n_total = sum(p.numel() for p in model._llada.parameters())
            maybe_print(f"[T6T7] LoRA ✓  r={args.lora_r} α={args.lora_alpha} "
                        f"targets={targets}  trainable={n_trainable/1e6:.2f}M / "
                        f"{n_total/1e6:.1f}M ({100*n_trainable/n_total:.3f}%)")
        else:
            # Full SFT path — ensure every param requires grad
            n_params = 0
            for p in model.parameters():
                p.requires_grad_(True)
                n_params += p.numel()
            maybe_print(f"[T6T7] full SFT: {n_params/1e6:.1f}M params trainable")

        # Gradient checkpointing (LLaDA remote code rejects; keep fail-soft)
        if args.gradient_checkpointing:
            inner = getattr(model, "_llada", None)
            # peft wraps — unwrap to reach HF model for GC call
            inner_hf = getattr(inner, "base_model", None) if args.use_lora else inner
            if inner_hf is not None and hasattr(inner_hf, "model"):
                inner_hf = inner_hf.model  # peft PeftModel.base_model.model
            target_for_gc = inner_hf if inner_hf is not None else inner
            if target_for_gc is not None and hasattr(target_for_gc, "gradient_checkpointing_enable"):
                try:
                    target_for_gc.gradient_checkpointing_enable()
                    if hasattr(target_for_gc, "config"):
                        try: target_for_gc.config.use_cache = False
                        except Exception: pass
                    maybe_print(f"[T6T7] gradient_checkpointing_enable() ✓")
                except ValueError as e:
                    maybe_print(f"[T6T7] GC unsupported ({e}); rely on "
                                f"FSDP/LoRA for memory")

        # ── Parallel wrap: FSDP (default for 8B) or DDP ──────────────────────
        if is_ddp:
            if args.parallel == "fsdp":
                import functools
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    MixedPrecision, ShardingStrategy,
                    StateDictType, FullStateDictConfig,
                )
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

                # FSDP wraps only the inner HF model — LLaDAWrapper stays
                # as the outer object. This keeps .noise_input /
                # .mask_token_id / .device / .tokenizer directly on model
                # (no TransparentDDP attr fallback needed). Finetuner sees
                # the wrapper; FSDP intercepts the inner forward/backward.
                mp = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
                wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=int(1e7),
                )
                model._llada = FSDP(
                    model._llada,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=mp,
                    auto_wrap_policy=wrap_policy,
                    device_id=torch.cuda.current_device(),
                    # use_orig_params=True is CRITICAL:
                    #   - LoRA has mixed frozen/trainable params
                    #   - Finetuner iterates model.parameters() for optim
                    #   - without it, FSDP flattens params and peft/optim
                    #     filtering by requires_grad breaks
                    use_orig_params=True,
                    sync_module_states=True,
                    limit_all_gathers=True,
                )
                # Configure state-dict behavior so HF export (below) +
                # Finetuner .state_dict() calls return the full consolidated
                # state on rank 0 only (avoids rank-local shards on disk).
                FSDP.set_state_dict_type(
                    model._llada,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                )
                maybe_print(f"[T6T7] FSDP FULL_SHARD ✓  "
                            f"rank={local_rank}  bf16 mixed-precision")
            else:  # ddp
                from torch.nn.parallel import DistributedDataParallel as DDP

                class TransparentDDP(DDP):
                    def __getattr__(self, name):
                        try: return super().__getattr__(name)
                        except AttributeError:
                            return getattr(self.module, name)

                model = TransparentDDP(model, device_ids=[local_rank],
                                       find_unused_parameters=False)
                maybe_print(f"[T6T7] DDP on cuda:{local_rank} "
                            f"(WARN: 8B+AdamW likely OOM without --use_lora)")

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
        keep_last_n=args.keep_last_n,
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

    # Rebuild optimizer to only track trainable params (LoRA filters most
    # out). CRITICAL: also rebuild scheduler — Finetuner.__init__ bound
    # scheduler to the *original* optimizer; replacing trainer.optimizer
    # alone leaves scheduler stepping the stale one (lr never updates on
    # the new one).
    if args.use_lora and hasattr(trainer, "optimizer"):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        trainable = [p for p in trainer.model.parameters() if p.requires_grad]
        trainer.optimizer = AdamW(trainable, lr=args.lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01)
        trainer.scheduler = CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=1000, T_mult=2,
        )
        maybe_print(f"[T6T7] LoRA: optimizer+scheduler rebuilt on "
                    f"{sum(p.numel() for p in trainable)/1e6:.2f}M params")

    # ── Resume from latest step_*.pt if present in save_dir ──────────────────
    # NOTE: Trainer.load_checkpoint uses plain load_state_dict — works for
    # DDP / single-GPU. Under FSDP it's unsafe: torch.load(map_location=
    # cuda:local_rank) stages the full state on one GPU (~16 GB bf16), and
    # load_state_dict into an FSDP-wrapped module without the correct
    # state_dict_type context gives rank-mismatched shapes.
    # For FSDP: refuse resume with a clear message rather than silent OOM.
    # (Proper FSDP resume would need FSDP.state_dict_type(FULL_STATE_DICT)
    # context + load_state_dict on all ranks — deferred to a follow-up.)
    latest_ckpt = None
    if save_dir.exists():
        step_ckpts = sorted(
            save_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0,
        )
        if step_ckpts:
            latest_ckpt = step_ckpts[-1]
    if latest_ckpt is not None and is_ddp and args.parallel == "fsdp":
        maybe_print(f"[T6T7] found {latest_ckpt.name} but FSDP resume is "
                    f"not yet implemented safely — starting fresh.")
        maybe_print(f"[T6T7]   (workaround: rerun with --parallel ddp "
                    f"--use_lora to resume; or delete {save_dir} and restart.)")
    elif latest_ckpt is not None and hasattr(trainer, "load_checkpoint"):
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

    # ── Export HF-format checkpoint for serving ─────────────────────────────
    # CRITICAL: under FSDP, `summon_full_params` and `state_dict()` are
    # COLLECTIVE calls — all ranks must enter together. Early-returning on
    # non-main ranks here would deadlock rank 0. So the gather context is
    # entered on ALL ranks; only rank 0 performs the actual disk write.
    #
    # Resolve the LLaDAWrapper:
    #   - fsdp path: model is still LLaDAWrapper (FSDP wrapped ._llada inner)
    #   - ddp path:  model is TransparentDDP(LLaDAWrapper), unwrap via .module
    #   - single-gpu: model is LLaDAWrapper directly
    if is_ddp and args.parallel == "ddp":
        inner_wrapper = model.module
    else:
        inner_wrapper = model

    hf_dir = save_dir / "hf"
    if is_main:
        hf_dir.mkdir(parents=True, exist_ok=True)

    try:
        inner = getattr(inner_wrapper, "_llada", None)
        if inner is None:
            raise AttributeError("LLaDAWrapper._llada missing")

        if is_ddp and args.parallel == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # summon_full_params is COLLECTIVE — all ranks enter. With
            # rank0_only=True, only rank 0 gets the unsharded tensors; the
            # others just participate in the all-gather then immediately
            # resume sharded state on context exit.
            with FSDP.summon_full_params(inner, writeback=False,
                                         offload_to_cpu=True,
                                         rank0_only=True):
                if is_main:
                    # Only rank 0 sees real params + writes
                    peft_or_hf = inner.module if hasattr(inner, "module") else inner
                    if args.use_lora and args.lora_merge_on_save:
                        merged = peft_or_hf.merge_and_unload()
                        merged.save_pretrained(hf_dir, safe_serialization=True)
                        print(f"[T6T7] FSDP+LoRA merged → {hf_dir}")
                    elif args.use_lora:
                        peft_or_hf.save_pretrained(hf_dir)
                        print(f"[T6T7] FSDP+LoRA adapter → {hf_dir}")
                    else:
                        peft_or_hf.save_pretrained(hf_dir, safe_serialization=True)
                        print(f"[T6T7] FSDP full-SFT → {hf_dir}")
            # Barrier so non-main ranks wait until rank 0 finishes the
            # final disk I/O + any post-processing below.
            import torch.distributed as dist
            dist.barrier()
        else:
            # ddp / single-gpu path — no collective gather needed
            if is_main:
                if args.use_lora and args.lora_merge_on_save:
                    merged = inner.merge_and_unload()
                    merged.save_pretrained(hf_dir, safe_serialization=True)
                    print(f"[T6T7] LoRA merged → {hf_dir}")
                else:
                    inner.save_pretrained(hf_dir, safe_serialization=True)

        # Tokenizer + trust_remote_code files — rank 0 only
        if is_main:
            if hasattr(inner_wrapper, "tokenizer"):
                inner_wrapper.tokenizer.save_pretrained(hf_dir)
            # Copy modeling_llada.py / configuration_llada.py etc. from the
            # source dir (if init_ckpt is a local path). HF-hub ids won't
            # resolve to a local dir, so this is a no-op there — user must
            # pass --init_ckpt <local_path> or ensure HF cache is populated.
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
        if is_main:
            print(f"[T6T7] WARN: HF export failed: {e}")
            print(f"[T6T7] .pt checkpoints still usable via load_checkpoint()")

    if is_main:
        print(f"[T6T7] done. checkpoints → {save_dir}")


if __name__ == "__main__":
    main()
