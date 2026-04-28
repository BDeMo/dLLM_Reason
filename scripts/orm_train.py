#!/usr/bin/env python
"""Train the ORM head on collected (prompt, output, label) data.

Frozen base model + tiny linear head. BCE-with-logits loss. Single GPU
fits comfortably; only the head's ~4k params are trained.

References:
  Cobbe et al. 2021 (arXiv:2110.14168)  — ORM verifier on gsm8k
  V-STaR (arXiv:2402.06457)               — pos+neg self-distill ORM

Usage:
  python scripts/orm_train.py \\
      --base_ckpt runs/training/v161_t6_ablate/hf_step_336 \\
      --train_jsonl runs/validation/orm_data/orm_train.jsonl \\
      --out_dir runs/training/orm_v1 \\
      --max_steps 2000 \\
      --batch_size 8 --lr 1e-4
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer

from dllm_reason.models.orm_head import ORMWrapper, ORMHead


class ORMDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 768):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        # Format: chat template prompt + assistant output
        msgs = [{"role": "user", "content": r["question"]}]
        prompt_text = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        full_text = prompt_text + r["output"]
        # tokenize prompt alone to know boundary
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        enc = self.tok(full_text, truncation=True,
                       max_length=self.max_seq_len, return_tensors="pt")
        L = enc["input_ids"].shape[1]
        # output_mask: 1 where token is in answer region (and non-pad)
        output_mask = torch.zeros(L, dtype=torch.long)
        if prompt_len < L:
            output_mask[prompt_len:] = 1
        # ensure we don't mark pad as output (no pad here pre-collate, but safe)
        output_mask = output_mask * enc["attention_mask"].squeeze(0)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "output_mask": output_mask,
            "label": float(r["label"]),
        }


def collate(batch, pad_id: int):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    out_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.float)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        attn[i, :L] = b["attention_mask"]
        out_mask[i, :L] = b["output_mask"]
        labels[i] = b["label"]
    return {"input_ids": input_ids, "attention_mask": attn,
            "output_mask": out_mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True,
                    help="HF model path (T6 ckpt). Frozen during training.")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default=None,
                    help="optional val split for early eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=768)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="head-only training; larger lr OK")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ── DDP init ────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda")
        world_size = 1
        rank = 0
    is_main = (rank == 0)

    def log(msg):
        if is_main:
            print(msg, flush=True)

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    if is_ddp:
        dist.barrier()

    if not is_ddp:
        print("[ORM-TRAIN] WARNING: LOCAL_RANK not set → running SINGLE-GPU. "
              "For 8-GPU DDP, launch via:\n"
              "  torchrun --standalone --nproc_per_node=8 scripts/orm_train.py ...",
              flush=True)
    log(f"[ORM-TRAIN] world_size={world_size}  loading base {args.base_ckpt} ...")
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, trust_remote_code=True)
    base = AutoModel.from_pretrained(args.base_ckpt, trust_remote_code=True,
                                     dtype=torch.bfloat16).to(device).eval()
    hidden_size = base.config.hidden_size

    model = ORMWrapper(base, hidden_size, pooling=args.pooling,
                       dropout=args.dropout, freeze_base=True).to(device)
    log(f"[ORM-TRAIN] head params: "
        f"{sum(p.numel() for p in model.head.parameters())/1e3:.1f}k")

    # Wrap only the head in DDP (base is frozen, no grads)
    if is_ddp:
        model.head = DDP(model.head, device_ids=[local_rank],
                         output_device=local_rank)
        head_params = model.head.module.parameters()
    else:
        head_params = model.head.parameters()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = ORMDataset(args.train_jsonl, tok, max_seq_len=args.max_seq_len)
    log(f"[ORM-TRAIN] train dataset: {len(ds)}")
    val_ds = (ORMDataset(args.val_jsonl, tok, max_seq_len=args.max_seq_len)
              if args.val_jsonl else None)

    if is_ddp:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed,
                                     drop_last=True)
        val_sampler = (DistributedSampler(val_ds, shuffle=False, drop_last=False)
                       if val_ds else None)
    else:
        sampler = None
        val_sampler = None

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                        num_workers=0,
                        collate_fn=lambda b: collate(b, pad_id))
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             sampler=val_sampler,
                             num_workers=0,
                             collate_fn=lambda b: collate(b, pad_id))
                  if val_ds else None)

    optim = torch.optim.AdamW(
        [p for p in head_params if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.max_steps,
    )
    bce = nn.BCEWithLogitsLoss()

    def head_state_dict():
        # unwrap DDP if needed
        return (model.head.module.state_dict() if is_ddp
                else model.head.state_dict())

    step = 0
    epoch = 0
    accum_loss = 0.0
    accum_acc = 0.0
    accum_n = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, attention_mask=attn,
                       output_mask=batch["output_mask"].to(device)).float()
        loss = bce(logits, labels)
        loss.backward()
        clip_params = (model.head.module.parameters() if is_ddp
                       else model.head.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        optim.step()
        scheduler.step()
        optim.zero_grad()

        accum_loss += loss.item()
        preds = (logits > 0).float()
        accum_acc += (preds == labels).float().sum().item()
        accum_n += labels.shape[0]
        step += 1

        if step % args.log_every == 0:
            # all-reduce metrics across ranks for accurate logs
            if is_ddp:
                t = torch.tensor([accum_loss, accum_acc, float(accum_n)],
                                 device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                tot_loss, tot_acc, tot_n = t.tolist()
                avg_loss = tot_loss / (args.log_every * world_size)
                avg_acc = tot_acc / max(tot_n, 1)
            else:
                avg_loss = accum_loss / args.log_every
                avg_acc = accum_acc / max(accum_n, 1)
            lr_now = scheduler.get_last_lr()[0]
            log(f"[ORM-TRAIN] step {step}: loss={avg_loss:.4f}  "
                f"acc={avg_acc:.3f}  lr={lr_now:.2e}")
            accum_loss = 0.0; accum_acc = 0.0; accum_n = 0

        if val_loader is not None and step % args.eval_every == 0:
            # only flip the head — base stays in eval (frozen, dropout off)
            (model.head.module if is_ddp else model.head).eval()
            v_loss = 0.0; v_acc = 0.0; v_n = 0
            with torch.no_grad():
                for vb in val_loader:
                    vl = model(vb["input_ids"].to(device),
                               attention_mask=vb["attention_mask"].to(device),
                               output_mask=vb["output_mask"].to(device)).float()
                    vt = vb["labels"].to(device)
                    v_loss += bce(vl, vt).item() * vt.shape[0]
                    v_acc += ((vl > 0).float() == vt).float().sum().item()
                    v_n += vt.shape[0]
            if is_ddp:
                t = torch.tensor([v_loss, v_acc, float(v_n)], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                v_loss, v_acc, v_n = t.tolist()
            log(f"[ORM-TRAIN] step {step}  val: "
                f"loss={v_loss/max(v_n,1):.4f}  acc={v_acc/max(v_n,1):.3f}")
            (model.head.module if is_ddp else model.head).train()

        if step % args.save_every == 0 and is_main:
            ckpt_path = out_dir / f"head_step_{step}.pt"
            torch.save(head_state_dict(), ckpt_path)
            log(f"[ORM-TRAIN] saved head → {ckpt_path}")

    # final save (rank 0 only)
    if is_main:
        final = out_dir / "head_final.pt"
        torch.save(head_state_dict(), final)
        log(f"[ORM-TRAIN] done. final → {final}")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
