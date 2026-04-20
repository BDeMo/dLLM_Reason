#!/usr/bin/env bash
# setup_env_v1.6.sh — Fresh Python environment for v1.6 (Qwen3.5-compatible).
#
# Builds a clean venv with:
#   - torch + CUDA wheels (matching the host's CUDA driver)
#   - transformers from master (required for Qwen3.5 architecture)
#   - accelerate / datasets / safetensors / huggingface_hub
#   - fastapi / uvicorn (serve.py)
#   - this repo installed in editable mode
#
# After this script finishes, activate with:
#   source .venv_v16/bin/activate
#
# Usage:
#   bash scripts/setup_env_v1.6.sh                  # defaults: .venv_v16, cuda=auto
#   bash scripts/setup_env_v1.6.sh --venv .venv_new
#   bash scripts/setup_env_v1.6.sh --cuda cu121     # pin cuda wheel index
#   bash scripts/setup_env_v1.6.sh --mirror         # use tsinghua pypi mirror
#   bash scripts/setup_env_v1.6.sh --no-transformers-master
#                                                   # use stable pypi transformers
#   bash scripts/setup_env_v1.6.sh --dry_run

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
VENV_DIR=".venv_v16"
CUDA_WHEEL=""            # auto-detect from nvidia-smi if empty; e.g. cu121 / cu124
USE_MIRROR=0
TRANSFORMERS_MASTER=1    # master needed for Qwen3.5; pass --no-transformers-master for stable
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)                       VENV_DIR="$2"; shift 2 ;;
        --cuda)                       CUDA_WHEEL="$2"; shift 2 ;;
        --mirror)                     USE_MIRROR=1; shift ;;
        --no-transformers-master)     TRANSFORMERS_MASTER=0; shift ;;
        --dry_run)                    DRY_RUN=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)
            echo "[SETUP] unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ── Auto-detect CUDA wheel tag ────────────────────────────────────────────────
if [[ -z "$CUDA_WHEEL" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CU_VER=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
        if [[ -n "$CU_VER" ]]; then
            # Map driver CUDA version to torch wheel tag. Rule of thumb:
            #   driver >= 12.4 → cu124, >= 12.1 → cu121, >= 11.8 → cu118
            # torch wheels are backward compatible up to a point.
            MAJOR=$(echo "$CU_VER" | cut -d. -f1)
            MINOR=$(echo "$CU_VER" | cut -d. -f2)
            if [[ "$MAJOR" -ge 12 ]]; then
                if [[ "$MINOR" -ge 4 ]]; then
                    CUDA_WHEEL="cu124"
                else
                    CUDA_WHEEL="cu121"
                fi
            else
                CUDA_WHEEL="cu118"
            fi
            echo "[SETUP] driver reports CUDA $CU_VER → torch wheel: $CUDA_WHEEL"
        fi
    fi
fi
[[ -z "$CUDA_WHEEL" ]] && CUDA_WHEEL="cu121"  # fallback

# ── Banner ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  v1.6 environment bootstrap"
echo "══════════════════════════════════════════════════════════════"
echo "  ROOT                   = $ROOT"
echo "  VENV_DIR               = $VENV_DIR"
echo "  CUDA_WHEEL             = $CUDA_WHEEL"
echo "  USE_MIRROR             = $USE_MIRROR (pypi tsinghua mirror)"
echo "  TRANSFORMERS_MASTER    = $TRANSFORMERS_MASTER"
echo "  DRY_RUN                = $DRY_RUN"
echo

# ── PIP index flags ───────────────────────────────────────────────────────────
PIP_ARGS=()
if [[ "$USE_MIRROR" -eq 1 ]]; then
    PIP_ARGS+=(-i "https://pypi.tuna.tsinghua.edu.cn/simple")
fi

run_or_dry() {
    local desc="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] $desc"
        echo "      $*"
        return 0
    fi
    echo "[RUN] $desc"
    echo "      $*"
    eval "$@"
}

# ── Step 1: create venv ──────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    echo "[SETUP] venv already exists at $VENV_DIR — skipping creation"
else
    run_or_dry "create venv" python -m venv "$VENV_DIR"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    # activate the venv for the remainder of the script
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    echo "[SETUP] active venv: $(which python)"
    echo "[SETUP] pip = $(which pip)"
fi

# ── Step 2: upgrade pip ──────────────────────────────────────────────────────
run_or_dry "upgrade pip" \
    python -m pip install --upgrade pip "${PIP_ARGS[@]}"

# ── Step 3: install torch (match CUDA) ───────────────────────────────────────
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_WHEEL"
run_or_dry "install torch (cuda=$CUDA_WHEEL)" \
    pip install torch torchvision torchaudio \
    --index-url "$TORCH_INDEX"

# ── Step 4: install transformers (master for Qwen3.5, else stable) ───────────
if [[ "$TRANSFORMERS_MASTER" -eq 1 ]]; then
    run_or_dry "install transformers from GitHub master (Qwen3.5 support)" \
        pip install "${PIP_ARGS[@]}" --upgrade \
        "git+https://github.com/huggingface/transformers.git"
else
    run_or_dry "install transformers (stable)" \
        pip install "${PIP_ARGS[@]}" --upgrade transformers
fi

# ── Step 5: install core runtime deps ────────────────────────────────────────
run_or_dry "install accelerate + datasets + hf_hub + safetensors" \
    pip install "${PIP_ARGS[@]}" --upgrade \
    accelerate \
    datasets \
    huggingface_hub \
    safetensors \
    tokenizers \
    sentencepiece \
    protobuf

# ── Step 6: install serving + validation deps ────────────────────────────────
run_or_dry "install serve + validation deps" \
    pip install "${PIP_ARGS[@]}" --upgrade \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    httpx \
    requests \
    bitsandbytes \
    hf_transfer

# ── Step 7: install this project in editable mode ───────────────────────────
if [[ -f "$ROOT/pyproject.toml" ]]; then
    run_or_dry "install dllm_reason in editable mode" \
        pip install "${PIP_ARGS[@]}" -e "$ROOT"
elif [[ -f "$ROOT/setup.py" ]]; then
    run_or_dry "install dllm_reason (legacy setup.py)" \
        pip install "${PIP_ARGS[@]}" -e "$ROOT"
else
    echo "[SETUP] WARN: no pyproject.toml / setup.py found — "
    echo "        dllm_reason not installed; add $ROOT/src to PYTHONPATH manually."
fi

# ── Step 8: verify critical imports + Qwen3.5 load path ──────────────────────
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo
    echo "══════════════════════════════════════════════════════════════"
    echo "  Verification"
    echo "══════════════════════════════════════════════════════════════"
    python - <<'PY'
import sys, traceback
ok = True

def check(label, fn):
    global ok
    try:
        fn()
        print(f"  ✓ {label}")
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        traceback.print_exc(limit=1)
        ok = False

def _torch_check():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"      torch={torch.__version__} cuda={torch.version.cuda} "
          f"#gpus={torch.cuda.device_count()}")

def _transformers_check():
    import transformers
    print(f"      transformers={transformers.__version__}")
    # See if Qwen3.5 architecture is registered
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    have_qwen35 = any("qwen3_5" in k.lower() or "qwen35" in k.lower()
                      for k in MODEL_MAPPING_NAMES)
    have_qwen3 = any(k.lower().startswith("qwen3") for k in MODEL_MAPPING_NAMES)
    print(f"      has qwen3 arch: {have_qwen3}  has qwen3_5 arch: {have_qwen35}")

def _accelerate_check():
    import accelerate
    print(f"      accelerate={accelerate.__version__}")

def _datasets_check():
    import datasets
    print(f"      datasets={datasets.__version__}")

def _hf_hub_check():
    import huggingface_hub
    print(f"      huggingface_hub={huggingface_hub.__version__}")

def _dllm_reason_check():
    import dllm_reason
    print(f"      dllm_reason loaded from {dllm_reason.__file__}")

def _fastapi_check():
    import fastapi
    print(f"      fastapi={fastapi.__version__}")

check("torch + CUDA", _torch_check)
check("transformers (Qwen3.5 arch presence)", _transformers_check)
check("accelerate", _accelerate_check)
check("datasets", _datasets_check)
check("huggingface_hub", _hf_hub_check)
check("dllm_reason (editable install)", _dllm_reason_check)
check("fastapi (serve.py)", _fastapi_check)

sys.exit(0 if ok else 1)
PY
    VERIFY_RC=$?

    # ── Step 9: freeze exact versions for reproducibility ───────────────────
    FREEZE_PATH="$ROOT/tmp/requirements_v16.txt"
    mkdir -p "$(dirname "$FREEZE_PATH")"
    pip freeze > "$FREEZE_PATH"
    echo "[SETUP] frozen requirements → $FREEZE_PATH"

    echo
    if [[ "$VERIFY_RC" -eq 0 ]]; then
        echo "[SETUP] ✓ ALL VERIFICATIONS PASSED"
    else
        echo "[SETUP] ✗ Some checks failed — see output above."
        echo "        venv is usable but you'll hit runtime errors on the failing deps."
    fi

    echo
    echo "Next steps:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python scripts/download_qwen.py --mirror hf-mirror --sizes 3.5-4B"
    echo "  bash scripts/run_v1.6.sh --mirror hf-mirror --t6_gpus 8"
    echo
fi
