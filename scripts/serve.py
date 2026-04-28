"""FastAPI inference server for dLLM-Reason.

Serves LLaDA model with configurable DAG strategies via REST API.
Supports hot-switching strategies without reloading the model.

Usage:
    python scripts/serve.py --model_id checkpoints/llada-instruct --port 8000
    python scripts/serve.py --model_id checkpoints/llada-instruct --quantize 4bit

API endpoints:
    POST /generate                  — generate text with a given strategy
    POST /batch_generate            — batch generate (multiple prompts, single strategy)
    POST /generate_span_revise      — A3: block denoise + sliding-window revise hook
    POST /generate_block_schedule   — A4: explicit per-block (size, steps) schedule
    POST /generate_bon              — P2.C: Best-of-N with ORM verifier (recommended default)
    POST /switch_model              — hot-swap the loaded model
    GET  /strategies                — list available strategies
    GET  /info                      — model info (id, device, dtype)
    GET  /health                    — health check
"""

import argparse
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="dLLM-Reason", version="1.6.0")

# ── Version note ─────────────────────────────────────────────────────────────
# Install serving extras: pip install "dllm-reason[serve]"
# CLI entry point:        dllm-serve --model_id checkpoints/llada-instruct
# ─────────────────────────────────────────────────────────────────────────────

# Global model reference (loaded at startup)
_model = None
_model_id = ""

# ORM verifier head (loaded if --orm_head passed at startup, or lazy-loaded
# from per-request override). See docs/archive/finding_p2c_orm_bon.md.
_orm_head = None
_orm_head_path = ""
_orm_pooling = "mean"


class GenerateRequest(BaseModel):
    prompt: str
    strategy: str = Field(default="confidence", description="Unmasking strategy")
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    num_steps: int = Field(default=128, ge=1, le=1024)
    block_length: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    cfg_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    remasking: str = Field(default="low_confidence")


class GenerateResponse(BaseModel):
    text: str
    strategy: str
    elapsed_seconds: float
    num_tokens: int


AVAILABLE_STRATEGIES = [
    "confidence", "random", "entropy", "semi_ar",
    "maskgit_cosine", "critical_token_first", "curriculum",
    "linear", "cot", "skeleton", "bidirectional", "answer_first",
    "adaptive_dynamic",
]


def build_scheduler(strategy: str, gen_len: int, block_length: int, device):
    """Build a scheduler from strategy name."""
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
    from dllm_reason.scheduler.random_scheduler import RandomScheduler
    from dllm_reason.scheduler.linear_scheduler import LinearScheduler
    from dllm_reason.scheduler.entropy_scheduler import EntropyScheduler
    from dllm_reason.scheduler.semi_ar_scheduler import SemiAutoregressiveScheduler
    from dllm_reason.scheduler.maskgit_scheduler import MaskGITCosineScheduler
    from dllm_reason.scheduler.critical_token_scheduler import CriticalTokenFirstScheduler
    from dllm_reason.scheduler.curriculum_scheduler import CurriculumScheduler
    from dllm_reason.scheduler.adaptive_dynamic_scheduler import AdaptiveDynamicScheduler

    schedulers = {
        "confidence": lambda: ConfidenceScheduler(),
        "random": lambda: RandomScheduler(),
        "linear": lambda: LinearScheduler(),
        "entropy": lambda: EntropyScheduler(),
        "semi_ar": lambda: SemiAutoregressiveScheduler(block_size=block_length),
        "maskgit_cosine": lambda: MaskGITCosineScheduler(),
        "critical_token_first": lambda: CriticalTokenFirstScheduler(),
        "curriculum": lambda: CurriculumScheduler(),
        "adaptive_dynamic": lambda: AdaptiveDynamicScheduler(),
    }

    if strategy in schedulers:
        return schedulers[strategy]()

    # DAG-based strategies
    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag, answer_first_dag,
    )

    if strategy == "cot":
        dag = chain_of_thought_dag(gen_len, num_steps=4, device=device)
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "skeleton":
        dag = skeleton_then_detail_dag(
            gen_len, list(range(0, gen_len, 3)), list(range(1, gen_len, 3)), device=device,
        )
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "bidirectional":
        dag = bidirectional_dag(gen_len, num_segments=4, device=device)
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "answer_first":
        dag = answer_first_dag(
            gen_len, list(range(int(gen_len * 0.8), gen_len)), device=device,
        )
        return DAGScheduler(dag, sub_strategy="confidence_topk")

    raise ValueError(f"Unknown strategy: {strategy}")


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_id, "device": str(_model.device if _model else "none")}


@app.get("/strategies")
def strategies():
    return {"strategies": AVAILABLE_STRATEGIES}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    if req.strategy not in AVAILABLE_STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy}. Available: {AVAILABLE_STRATEGIES}")

    scheduler = build_scheduler(req.strategy, req.max_new_tokens, req.block_length, _model.device)

    t0 = time.time()
    text = _model.generate(
        prompt=req.prompt,
        generation_len=req.max_new_tokens,
        block_length=req.block_length,
        scheduler=scheduler,
        num_steps=req.num_steps,
        temperature=req.temperature,
        cfg_scale=req.cfg_scale,
        remasking=req.remasking,
        system_prompt=req.system_prompt,
    )
    elapsed = time.time() - t0

    return GenerateResponse(
        text=text,
        strategy=req.strategy,
        elapsed_seconds=round(elapsed, 3),
        num_tokens=len(text.split()),
    )


# ── Batch generation ─────────────────────────────────────────────────────────


class BatchGenerateRequest(BaseModel):
    prompts: list[str]
    strategy: str = Field(default="confidence", description="Unmasking strategy")
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    num_steps: int = Field(default=128, ge=1, le=1024)
    block_length: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    cfg_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    remasking: str = Field(default="low_confidence")


@app.post("/batch_generate", response_model=list[GenerateResponse])
def batch_generate(req: BatchGenerateRequest):
    """Generate text for multiple prompts with a single strategy.

    Currently iterates prompts sequentially (each prompt gets its own
    scheduler instance). True batch inference (stacking into a single
    (B, L) tensor) can be added later as an optimisation.
    """
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    if req.strategy not in AVAILABLE_STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy}. Available: {AVAILABLE_STRATEGIES}")

    results: list[GenerateResponse] = []
    for prompt in req.prompts:
        scheduler = build_scheduler(
            req.strategy, req.max_new_tokens, req.block_length, _model.device,
        )
        t0 = time.time()
        text = _model.generate(
            prompt=prompt,
            generation_len=req.max_new_tokens,
            block_length=req.block_length,
            scheduler=scheduler,
            num_steps=req.num_steps,
            temperature=req.temperature,
            cfg_scale=req.cfg_scale,
            remasking=req.remasking,
        )
        elapsed = time.time() - t0
        results.append(GenerateResponse(
            text=text,
            strategy=req.strategy,
            elapsed_seconds=round(elapsed, 3),
            num_tokens=len(text.split()),
        ))
    return results


# ── Validation-axis custom loops (A3 span-revise, A4 non-uniform blocks) ─────


class SpanReviseRequest(BaseModel):
    prompt: str
    gen_length: int = Field(default=128, ge=1, le=2048)
    steps: int = Field(default=128, ge=1, le=1024)
    block_length: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    revise_every: int = Field(default=8, ge=0, le=1024)
    revise_thresh: float = Field(default=0.4, ge=0.0, le=1.0)
    window_size: int = Field(default=4, ge=1, le=64)


@app.post("/generate_span_revise", response_model=GenerateResponse)
def generate_span_revise_endpoint(req: SpanReviseRequest):
    """A3 experiment: block-wise denoising + sliding-window mean-conf revise hook."""
    if _model is None:
        raise HTTPException(500, "Model not loaded")

    from dllm_reason.inference.validation_ext import generate_span_revise

    t0 = time.time()
    text = generate_span_revise(
        _model._llada, _model.tokenizer, req.prompt,
        gen_length=req.gen_length, steps=req.steps,
        block_length=req.block_length, temperature=req.temperature,
        revise_every=req.revise_every, revise_thresh=req.revise_thresh,
        window_size=req.window_size,
        mask_id=_model.mask_token_id,
    )
    return GenerateResponse(
        text=text,
        strategy=f"span_revise(w={req.window_size},thr={req.revise_thresh},every={req.revise_every})",
        elapsed_seconds=round(time.time() - t0, 3),
        num_tokens=len(text.split()),
    )


class BlockScheduleRequest(BaseModel):
    prompt: str
    block_sizes: list[int] = Field(description="e.g. [16, 16, 16, 16, 64]")
    steps_per_block: list[int] = Field(description="e.g. [16, 16, 16, 16, 64]")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


@app.post("/generate_block_schedule", response_model=GenerateResponse)
def generate_block_schedule_endpoint(req: BlockScheduleRequest):
    """A4 experiment: explicit per-block (size, steps) schedule for non-uniform layouts."""
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    if len(req.block_sizes) != len(req.steps_per_block):
        raise HTTPException(400, "block_sizes and steps_per_block must be same length")

    from dllm_reason.inference.validation_ext import generate_block_schedule

    t0 = time.time()
    text = generate_block_schedule(
        _model._llada, _model.tokenizer, req.prompt,
        block_sizes=req.block_sizes,
        steps_per_block=req.steps_per_block,
        temperature=req.temperature,
        mask_id=_model.mask_token_id,
    )
    return GenerateResponse(
        text=text,
        strategy=f"block_schedule({req.block_sizes})",
        elapsed_seconds=round(time.time() - t0, 3),
        num_tokens=len(text.split()),
    )


class InpaintAnchor(BaseModel):
    start_pos: int = Field(ge=0, description="start position inside gen region")
    text: str = Field(description="anchor text to pre-commit at this position")


class InpaintRequest(BaseModel):
    prompt: str
    anchors: list[InpaintAnchor] = Field(
        description="list of (start_pos, text) pairs; anchors are pre-committed "
                    "inside the gen region, later ones override earlier ones on overlap"
    )
    gen_length: int = Field(default=128, ge=1, le=2048)
    steps: int = Field(default=128, ge=1, le=1024)
    block_length: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


@app.post("/generate_inpaint", response_model=GenerateResponse)
def generate_inpaint_endpoint(req: InpaintRequest):
    """Strategy-search: inpainting with anchor tokens pre-committed inside gen region.

    Realises ``template_position`` ∈ {prefix, suffix, mid, scaffold} by placing
    a template span at arbitrary positions in the generation region instead of
    only at the prompt prefix. Supports pass@N via stochastic temperature.
    """
    if _model is None:
        raise HTTPException(500, "Model not loaded")

    from dllm_reason.inference.validation_ext import generate_inpaint

    anchors = [(a.start_pos, a.text) for a in req.anchors]
    t0 = time.time()
    text = generate_inpaint(
        _model._llada, _model.tokenizer, req.prompt,
        anchors=anchors,
        gen_length=req.gen_length, steps=req.steps,
        block_length=req.block_length, temperature=req.temperature,
        mask_id=_model.mask_token_id,
    )
    return GenerateResponse(
        text=text,
        strategy=f"inpaint(n_anchors={len(anchors)})",
        elapsed_seconds=round(time.time() - t0, 3),
        num_tokens=len(text.split()),
    )


# ── P2.C: Best-of-N with ORM verifier ────────────────────────────────────────
#
# Recommended default decode strategy (see docs/archive/finding_p2c_orm_bon.md).
# Result on T6/gsm8k:
#   greedy 33.8% / SC@8 39.6% / BoN@8 49.2% / pass@8 65.0% (oracle)
#
# IMPORTANT: the ORM head is base-model-specific. If you swap the base
# (different ckpt / different SFT / different architecture), retrain via:
#   bash scripts/orm_pipeline.sh --base_ckpt <new_ckpt> --pooling mean
# Pipeline is ~1h end-to-end on 8 GPU. Without --orm_head, this endpoint
# falls back to SC@N (majority vote) — still beats greedy by ~5pp.


def _ensure_orm_head(head_path: str | None = None, pooling: str | None = None):
    """Load ORM head on demand. Returns the head module or None if unavailable."""
    global _orm_head, _orm_head_path, _orm_pooling
    target_path = head_path or _orm_head_path
    target_pool = pooling or _orm_pooling
    if not target_path:
        return None
    if _orm_head is not None and target_path == _orm_head_path \
            and target_pool == _orm_pooling:
        return _orm_head

    import sys
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from dllm_reason.models.orm_head import ORMHead

    hidden_size = _model._llada.config.hidden_size
    head = ORMHead(hidden_size, pooling=target_pool)
    head.load_state_dict(torch.load(target_path, map_location=_model.device))
    head.to(_model.device).eval()
    _orm_head = head
    _orm_head_path = target_path
    _orm_pooling = target_pool
    print(f"[BoN] ORM head loaded: {target_path}  (pooling={target_pool})")
    return _orm_head


def _bon_score(prompts: list[str], outputs: list[str]) -> list[float]:
    """Score (prompt, output) pairs with the loaded ORM head."""
    tok = _model.tokenizer
    base = _model._llada
    texts, prompt_lens = [], []
    for p, o in zip(prompts, outputs):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False)
        prompt_lens.append(len(tok(chat, add_special_tokens=False)["input_ids"]))
        texts.append(chat + o)
    enc = tok(texts, padding=True, truncation=True, max_length=768,
              return_tensors="pt").to(_model.device)
    L = enc["input_ids"].shape[1]
    out_mask = torch.zeros_like(enc["attention_mask"])
    for i, pl in enumerate(prompt_lens):
        if pl < L:
            out_mask[i, pl:] = 1
    out_mask = out_mask * enc["attention_mask"]
    with torch.no_grad():
        try:
            base_out = base(enc["input_ids"], attention_mask=enc["attention_mask"],
                            output_hidden_states=True)
        except TypeError:
            base_out = base(enc["input_ids"], output_hidden_states=True)
        hidden = (base_out.hidden_states[-1]
                  if hasattr(base_out, "hidden_states") and base_out.hidden_states
                  else base_out.last_hidden_state)
        logits = _orm_head(hidden, attention_mask=enc["attention_mask"],
                           output_mask=out_mask).float()
    return logits.tolist()


class BoNRequest(BaseModel):
    prompt: str
    n_samples: int = Field(default=8, ge=1, le=32)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0,
                               description="must be > 0 for sampling diversity")
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    num_steps: int = Field(default=128, ge=1, le=1024)
    block_length: int = Field(default=32, ge=1, le=512)
    strategy: str = Field(default="confidence")
    orm_head: str | None = Field(
        default=None,
        description="optional override; if no head loaded and not provided, "
                    "falls back to SC@N (majority vote)")
    pooling: str = Field(default="mean", description='"mean" or "last"')
    return_all: bool = Field(default=False,
                             description="if true, return all N samples + scores")


class BoNResponse(BaseModel):
    text: str
    selected_idx: int
    method: str  # "BoN" or "SC@N (no verifier)"
    n_samples: int
    elapsed_seconds: float
    samples: list[str] | None = None
    scores: list[float] | None = None


@app.post("/generate_bon", response_model=BoNResponse)
def generate_bon_endpoint(req: BoNRequest):
    """Best-of-N with ORM verifier (recommended default).

    Samples N traces at the requested temperature, scores each with the
    loaded ORM head, returns the argmax. Falls back to SC@N (majority
    vote on extracted answer) if no ORM head is loaded.
    """
    import re
    from collections import Counter

    if _model is None:
        raise HTTPException(500, "Model not loaded")
    if req.temperature <= 0:
        raise HTTPException(400, "BoN needs temperature > 0 for diversity")

    head = _ensure_orm_head(req.orm_head, req.pooling)

    # Sample N
    t0 = time.time()
    samples: list[str] = []
    for _ in range(req.n_samples):
        scheduler = build_scheduler(
            req.strategy, req.max_new_tokens, req.block_length, _model.device)
        text = _model.generate(
            prompt=req.prompt,
            generation_len=req.max_new_tokens,
            block_length=req.block_length,
            scheduler=scheduler,
            num_steps=req.num_steps,
            temperature=req.temperature,
        )
        samples.append(text)

    # Score
    if head is not None:
        scores = _bon_score([req.prompt] * req.n_samples, samples)
        best_i = int(max(range(req.n_samples), key=lambda j: scores[j]))
        method = "BoN"
    else:
        # SC@N fallback: majority vote on last numeric span
        def extract(s):
            nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", str(s or ""))
            return nums[-1].replace(",", "") if nums else None
        ans = [extract(s) for s in samples]
        ans_clean = [a for a in ans if a is not None]
        if ans_clean:
            mode_a = Counter(ans_clean).most_common(1)[0][0]
            best_i = next(i for i, a in enumerate(ans) if a == mode_a)
        else:
            best_i = 0
        scores = None
        method = "SC@N (no verifier)"

    return BoNResponse(
        text=samples[best_i],
        selected_idx=best_i,
        method=method,
        n_samples=req.n_samples,
        elapsed_seconds=round(time.time() - t0, 3),
        samples=samples if req.return_all else None,
        scores=scores if req.return_all else None,
    )


# ── Model hot-swap ───────────────────────────────────────────────────────────


class SwitchModelRequest(BaseModel):
    model_id: str
    torch_dtype: str = Field(default="bfloat16")
    quantize: str | None = Field(default=None, description="4bit or 8bit")


@app.post("/switch_model")
def switch_model(req: SwitchModelRequest):
    """Hot-swap the loaded model (e.g. after fine-tuning Stage 3)."""
    global _model, _model_id

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    # Free current model + invalidate ORM head (head is base-specific)
    global _orm_head, _orm_head_path
    if _model is not None:
        del _model
        torch.cuda.empty_cache()
    if _orm_head is not None:
        print("[BoN] base swapped → invalidating ORM head "
              "(retrain via scripts/orm_pipeline.sh for the new base)")
        _orm_head = None
        _orm_head_path = ""

    quant_config = None
    if req.quantize == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map.get(req.torch_dtype, torch.bfloat16),
            bnb_4bit_quant_type="nf4",
        )
    elif req.quantize == "8bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    from dllm_reason.models.llada import LLaDAWrapper
    _model = LLaDAWrapper(
        model_id=req.model_id,
        torch_dtype=dtype_map.get(req.torch_dtype, torch.bfloat16),
        device_map="auto",
        quantization_config=quant_config,
    )
    _model_id = req.model_id

    return {
        "status": "ok",
        "model_id": _model_id,
        "device": str(_model.device),
    }


# ── Model info ───────────────────────────────────────────────────────────────


@app.get("/info")
def info():
    """Return current model metadata (used by pipeline for health check)."""
    if _model is None:
        return {"status": "no_model", "model_id": None}
    return {
        "status": "ready",
        "model_id": _model_id,
        "device": str(_model.device),
        "dtype": str(getattr(_model, "torch_dtype", "unknown")),
    }


def main():
    global _model, _model_id

    parser = argparse.ArgumentParser(description="dLLM-Reason inference server")
    parser.add_argument("--model_id", type=str, default="checkpoints/llada-instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["4bit", "8bit"],
                        help="Load model with quantization (requires bitsandbytes)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--orm_head", type=str, default=None,
                        help="Path to ORM head .pt for /generate_bon. "
                             "Must be trained on the same base "
                             "(scripts/orm_pipeline.sh). If omitted, "
                             "/generate_bon falls back to SC@N.")
    parser.add_argument("--orm_pooling", type=str, default="mean",
                        choices=["mean", "last"],
                        help='ORM pooling — "mean" recommended for LLaDA')
    args = parser.parse_args()

    _model_id = args.model_id
    global _orm_head_path, _orm_pooling
    _orm_head_path = args.orm_head or ""
    _orm_pooling = args.orm_pooling

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    print(f"Loading model: {args.model_id}")

    # Build quantization config if requested
    quant_config = None
    if args.quantize == "4bit":
        print("Loading with 4-bit quantization (bitsandbytes)")
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map[args.torch_dtype],
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantize == "8bit":
        print("Loading with 8-bit quantization (bitsandbytes)")
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    from dllm_reason.models.llada import LLaDAWrapper
    _model = LLaDAWrapper(
        model_id=args.model_id,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        quantization_config=quant_config,
    )

    print(f"Model loaded on {_model.device}, serving at {args.host}:{args.port}")
    if args.orm_head:
        try:
            _ensure_orm_head(args.orm_head, args.orm_pooling)
        except Exception as e:
            print(f"[BoN] WARNING: failed to load ORM head: {e}\n"
                  f"      /generate_bon will fall back to SC@N.")
    else:
        print("[BoN] no --orm_head set; /generate_bon will fall back to SC@N. "
              "To enable BoN, train via scripts/orm_pipeline.sh.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
