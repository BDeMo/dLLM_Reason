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
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

from dllm_reason.models.orm_head import ORMWrapper


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
        enc = self.tok(full_text, truncation=True,
                       max_length=self.max_seq_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": float(r["label"]),
        }


def collate(batch, pad_id: int):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.float)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        attn[i, :L] = b["attention_mask"]
        labels[i] = b["label"]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    print(f"[ORM-TRAIN] loading base {args.base_ckpt} ...")
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, trust_remote_code=True)
    base = AutoModel.from_pretrained(args.base_ckpt, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16).cuda().eval()
    hidden_size = base.config.hidden_size

    model = ORMWrapper(base, hidden_size, pooling=args.pooling,
                       dropout=args.dropout, freeze_base=True).cuda()
    print(f"[ORM-TRAIN] head params: "
          f"{sum(p.numel() for p in model.head.parameters())/1e3:.1f}k")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = ORMDataset(args.train_jsonl, tok, max_seq_len=args.max_seq_len)
    print(f"[ORM-TRAIN] train dataset: {len(ds)}")
    val_ds = (ORMDataset(args.val_jsonl, tok, max_seq_len=args.max_seq_len)
              if args.val_jsonl else None)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0,
                        collate_fn=lambda b: collate(b, pad_id))
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0,
                             collate_fn=lambda b: collate(b, pad_id))
                  if val_ds else None)

    optim = torch.optim.AdamW(
        [p for p in model.head.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.max_steps,
    )
    bce = nn.BCEWithLogitsLoss()

    step = 0
    accum_loss = 0.0
    accum_acc = 0.0
    accum_n = 0
    data_iter = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        ids = batch["input_ids"].cuda()
        attn = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        logits = model(ids, attention_mask=attn).float()
        loss = bce(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        optim.step()
        scheduler.step()
        optim.zero_grad()

        accum_loss += loss.item()
        preds = (logits > 0).float()
        accum_acc += (preds == labels).float().sum().item()
        accum_n += labels.shape[0]
        step += 1

        if step % args.log_every == 0:
            avg_loss = accum_loss / args.log_every
            avg_acc = accum_acc / max(accum_n, 1)
            lr_now = scheduler.get_last_lr()[0]
            print(f"[ORM-TRAIN] step {step}: loss={avg_loss:.4f}  "
                  f"acc={avg_acc:.3f}  lr={lr_now:.2e}")
            accum_loss = 0.0; accum_acc = 0.0; accum_n = 0

        if val_loader is not None and step % args.eval_every == 0:
            model.eval()
            v_loss = 0.0; v_acc = 0.0; v_n = 0
            with torch.no_grad():
                for vb in val_loader:
                    vl = model(vb["input_ids"].cuda(),
                               attention_mask=vb["attention_mask"].cuda()).float()
                    vt = vb["labels"].cuda()
                    v_loss += bce(vl, vt).item() * vt.shape[0]
                    v_acc += ((vl > 0).float() == vt).float().sum().item()
                    v_n += vt.shape[0]
            print(f"[ORM-TRAIN] step {step}  val: "
                  f"loss={v_loss/max(v_n,1):.4f}  acc={v_acc/max(v_n,1):.3f}")
            model.train()

        if step % args.save_every == 0:
            ckpt_path = out_dir / f"head_step_{step}.pt"
            torch.save(model.head.state_dict(), ckpt_path)
            print(f"[ORM-TRAIN] saved head → {ckpt_path}")

    # final save
    final = out_dir / "head_final.pt"
    torch.save(model.head.state_dict(), final)
    print(f"[ORM-TRAIN] done. final → {final}")


if __name__ == "__main__":
    main()
