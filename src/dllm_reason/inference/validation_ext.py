"""Extended sampling loops used by the A-axis validation experiments.

Lives inside the package (not under ``scripts/validate/``) so the FastAPI
server (``scripts/serve.py``) can reuse the exact same generate() code that
the local reference scripts used. This is what lets A3/A4/A5 clients hit
the server via HTTP instead of loading LLaDA locally.

Two custom loops:

* :func:`generate_span_revise` — sliding-window mean-confidence revise hook
  (A3 experiment).
* :func:`generate_block_schedule` — arbitrary per-block ``(size, steps)``
  schedule, so A4 can run non-uniform layouts like ``short-then-long``.

Both expect a raw HuggingFace LLaDA model (``_llada`` attribute on
``LLaDAWrapper``) and the matching tokenizer. Neither function is on the
hot path for training — they're inference-only utilities.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _add_gumbel(logits: torch.Tensor, t: float) -> torch.Tensor:
    if t == 0:
        return logits
    return logits + t * torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)


def _num_transfer(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    B = mask_index.shape[0]
    n_mask = mask_index.sum(dim=1)
    base = n_mask // steps
    extra = n_mask % steps
    sched = base.unsqueeze(1).expand(B, steps).clone()
    for b in range(B):
        sched[b, : extra[b]] += 1
    return sched


def _encode_prompt(tokenizer, prompt: str, device: torch.device) -> tuple[torch.Tensor, int]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    return input_ids, input_ids.shape[1]


@torch.no_grad()
def generate_span_revise(
    model,
    tokenizer,
    prompt: str,
    *,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    revise_every: int = 0,
    revise_thresh: float = 0.4,
    window_size: int = 4,
    mask_id: int,
) -> str:
    """Block-wise denoising + sliding-window span revise hook.

    Every ``revise_every`` global steps, recompute a 1-D moving average of
    committed-token confidences (window = ``window_size``). For any window
    whose mean is below ``revise_thresh`` AND has enough committed tokens
    to be reliable, mask **all** committed positions inside the window.
    """
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    device = next(model.parameters()).device

    input_ids, prompt_len = _encode_prompt(tokenizer, prompt, device)

    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    committed_conf = torch.full_like(x, float("inf"), dtype=torch.float)
    committed_conf[:, :prompt_len] = float("inf")

    global_step = 0
    for block_idx in range(num_blocks):
        b_start = prompt_len + block_idx * block_length
        b_end = prompt_len + (block_idx + 1) * block_length
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, b_start:b_end] = True
        block_masked = x[:, b_start:b_end] == mask_id
        n_xfer = _num_transfer(block_masked, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x).logits
            x0 = _add_gumbel(logits, temperature).argmax(dim=-1)
            p = F.softmax(logits.double(), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1).float()
            x0_p = x0_p.masked_fill(~block_mask, -float("inf"))
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p,
                               torch.full_like(x0_p, -float("inf")))
            n = int(n_xfer[0, step].item())
            if n > 0:
                _, top_idx = torch.topk(conf[0], k=n)
                transfer = torch.zeros_like(x, dtype=torch.bool)
                transfer[0, top_idx] = True
                x[transfer] = x0[transfer]
                committed_conf[transfer] = torch.gather(
                    F.softmax(logits.float(), -1)[0], -1, x0[0].unsqueeze(-1)
                ).squeeze(-1)[transfer[0]]

            global_step += 1
            if revise_every > 0 and global_step % revise_every == 0:
                committed = (x != mask_id)
                committed[:, :prompt_len] = False
                committed[:, b_end:] = False
                vals = committed_conf[0]
                is_val = committed[0].float()
                vals_z = torch.where(committed[0], vals, torch.zeros_like(vals))
                kern = torch.ones(window_size, device=device)
                pad = window_size // 2
                sums = F.conv1d(vals_z.view(1, 1, -1),
                                kern.view(1, 1, -1), padding=pad).view(-1)[:vals.shape[0]]
                counts = F.conv1d(is_val.view(1, 1, -1),
                                  kern.view(1, 1, -1), padding=pad).view(-1)[:vals.shape[0]]
                mean = sums / counts.clamp(min=1.0)
                reliable = counts >= max(2, window_size // 2)
                bad_center = reliable & (mean < revise_thresh) & committed[0]
                if bad_center.any():
                    bad_mask = F.conv1d(bad_center.float().view(1, 1, -1),
                                        kern.view(1, 1, -1), padding=pad).view(-1)[:vals.shape[0]] > 0
                    kill = bad_mask & committed[0]
                    if kill.any():
                        x[0, kill] = mask_id
                        committed_conf[0, kill] = float("inf")

    return tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def generate_block_schedule(
    model,
    tokenizer,
    prompt: str,
    *,
    block_sizes: Sequence[int],
    steps_per_block: Sequence[int],
    temperature: float = 0.0,
    mask_id: int,
) -> str:
    """Denoise with an explicit (block_size, steps) schedule.

    Each block is denoised independently with its own step budget, so A4
    can evaluate e.g. ``block_sizes=[16,16,16,16,64]`` (short-then-long).
    Total ``gen_length = sum(block_sizes)``.
    """
    assert len(block_sizes) == len(steps_per_block)
    gen_length = sum(block_sizes)
    device = next(model.parameters()).device

    input_ids, prompt_len = _encode_prompt(tokenizer, prompt, device)

    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    off = prompt_len
    for blk_sz, n_step in zip(block_sizes, steps_per_block):
        b_start = off
        b_end = off + blk_sz
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, b_start:b_end] = True
        block_masked = x[:, b_start:b_end] == mask_id
        n_xfer = _num_transfer(block_masked, n_step)

        for step in range(n_step):
            mask_index = (x == mask_id)
            logits = model(x).logits
            x0 = _add_gumbel(logits, temperature).argmax(dim=-1)
            p = F.softmax(logits.double(), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1).float()
            x0_p = x0_p.masked_fill(~block_mask, -float("inf"))
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p,
                               torch.full_like(x0_p, -float("inf")))
            n = int(n_xfer[0, step].item())
            if n > 0:
                _, top_idx = torch.topk(conf[0], k=n)
                transfer = torch.zeros_like(x, dtype=torch.bool)
                transfer[0, top_idx] = True
                x[transfer] = x0[transfer]
        off = b_end

    return tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def generate_uniform(
    model,
    tokenizer,
    prompt: str,
    *,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int,
) -> str:
    """Plain block-wise denoising (no revise hook). Equivalent to h1.generate
    with ``revise_every=0`` — used as baseline inside A3."""
    num_blocks = gen_length // block_length
    return generate_block_schedule(
        model, tokenizer, prompt,
        block_sizes=[block_length] * num_blocks,
        steps_per_block=[steps // num_blocks] * num_blocks,
        temperature=temperature,
        mask_id=mask_id,
    )


@torch.no_grad()
def generate_inpaint(
    model,
    tokenizer,
    prompt: str,
    *,
    anchors: Sequence[tuple[int, str]],  # list of (start_pos_in_gen, anchor_text)
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int,
) -> str:
    """Block-wise denoising with **inpainting** — anchor tokens pre-committed
    at user-specified positions inside the generation region.

    Each anchor (start_pos, text) is tokenized and written into
    ``x[:, prompt_len+start_pos : prompt_len+start_pos+len(anchor_ids)]``
    **before** sampling begins. Because those positions are no longer equal
    to ``mask_id``, the standard block-wise unmasking loop skips them
    automatically — so we don't need an explicit lock mask.

    Used by strategy_search to realise ``template_position`` ∈
    {prefix, suffix, mid, scaffold}: the same template text is placed at
    different positions inside the gen region instead of only at the
    prompt prefix.

    Overlap policy: later anchors override earlier ones (last-write-wins).
    Anchors extending past ``gen_length`` are truncated with a warning.
    """
    assert gen_length % block_length == 0, \
        f"gen_length={gen_length} must be divisible by block_length={block_length}"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, \
        f"steps={steps} must be divisible by num_blocks={num_blocks}"
    device = next(model.parameters()).device

    input_ids, prompt_len = _encode_prompt(tokenizer, prompt, device)

    # Initialise sequence: [prompt | mask × gen_length]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    # Write each anchor into the gen region (relative to prompt_len)
    anchor_positions_gen: list[tuple[int, int]] = []  # (start, end) in gen region
    for start_pos, anchor_text in anchors:
        if not anchor_text:
            continue
        # tokenize without special tokens so we only get the anchor chars
        anchor_ids = tokenizer(anchor_text, add_special_tokens=False,
                               return_tensors="pt")["input_ids"].to(device)
        anchor_ids = anchor_ids[0]  # (L,)
        end_pos = start_pos + anchor_ids.shape[0]
        if end_pos > gen_length:
            import warnings
            warnings.warn(
                f"anchor '{anchor_text[:40]}...' at start={start_pos} "
                f"overflows gen_length={gen_length}; truncating to fit"
            )
            anchor_ids = anchor_ids[: gen_length - start_pos]
            end_pos = gen_length
        if start_pos < 0 or anchor_ids.shape[0] == 0:
            continue
        x[0, prompt_len + start_pos : prompt_len + end_pos] = anchor_ids
        anchor_positions_gen.append((start_pos, end_pos))

    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        b_start = prompt_len + block_idx * block_length
        b_end = prompt_len + (block_idx + 1) * block_length

        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, b_start:b_end] = True

        # Positions that are still masked inside this block — anchors pre-committed
        # above are already non-mask, so they get naturally excluded here.
        block_masked = x[:, b_start:b_end] == mask_id
        if not block_masked.any():
            # Whole block filled by anchors — skip denoising this block entirely
            continue
        n_xfer = _num_transfer(block_masked, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x).logits
            x0 = _add_gumbel(logits, temperature).argmax(dim=-1)
            p = F.softmax(logits.double(), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1).float()
            x0_p = x0_p.masked_fill(~block_mask, -float("inf"))
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p,
                               torch.full_like(x0_p, -float("inf")))
            n = int(n_xfer[0, step].item())
            if n > 0:
                _, top_idx = torch.topk(conf[0], k=n)
                transfer = torch.zeros_like(x, dtype=torch.bool)
                transfer[0, top_idx] = True
                x[transfer] = x0[transfer]

    return tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)
