"""Self-heal LLaDA checkpoint `modeling_llada.py` for transformers >= 5.x.

The upstream `GSAI-ML/LLaDA-8B-*` checkpoints ship a `modeling_llada.py`
that predates several transformers 5.x API changes:

1. `tie_weights(self)` — transformers 5.x passes new kwargs
   (`missing_keys`, `recompute_mapping`) during loading.
2. `self.config.use_cache` / `self.config.use_return_dict` — transformers
   5.x no longer auto-provides these on PretrainedConfig.

Those attributes cannot be fixed via pure monkey-patching before
`AutoModel.from_pretrained` returns (the call chain reads them during
loading and before the model is instantiated). Instead we rewrite the
local checkpoint copy of `modeling_llada.py` in place, idempotently.

This file is pure stdlib, safe to import, and does nothing if the
checkpoint is already patched or does not exist.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)

_SENTINEL = "# dllm_reason:llada-compat-patched"

_REPLACEMENTS = [
    (
        "    def tie_weights(self):\n"
        "        if self.config.weight_tying:",
        "    def tie_weights(self, *args, **kwargs):\n"
        "        # Accept & ignore transformers >= 5.x kwargs.\n"
        "        if self.config.weight_tying:",
    ),
    (
        "        if use_cache is None:\n"
        "            use_cache = self.config.use_cache",
        "        if use_cache is None:\n"
        '            use_cache = getattr(self.config, "use_cache", False)',
    ),
    (
        "        return_dict = return_dict if return_dict is not None else self.config.use_return_dict",
        '        return_dict = return_dict if return_dict is not None else getattr(\n'
        '            self.config, "use_return_dict", True\n'
        '        )',
    ),
]


def _clear_hf_dynamic_module_cache() -> None:
    """Remove the HF cached copy so the patched file is re-imported next load."""
    cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if not cache_root.exists():
        return
    for d in cache_root.glob("llada*"):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def ensure_llada_checkpoint_patched(checkpoint_path: str | Path) -> bool:
    """Patch `modeling_llada.py` in a LLaDA checkpoint directory if needed.

    Idempotent: safe to call repeatedly.

    Returns True if a patch was applied in this call (False if the file
    was missing or already patched).
    """
    ckpt = Path(checkpoint_path)
    target = ckpt / "modeling_llada.py"
    if not target.is_file():
        return False

    src = target.read_text()
    if _SENTINEL in src:
        return False

    patched = src
    applied = 0
    already = 0
    for old, new in _REPLACEMENTS:
        if old in patched:
            patched = patched.replace(old, new, 1)
            applied += 1
        elif new in patched:
            already += 1

    # If nothing matched at all, this is an unfamiliar modeling file;
    # don't touch it.
    if applied == 0 and already == 0:
        return False

    # Prepend sentinel so subsequent runs skip quickly.
    patched = f"{_SENTINEL}\n{patched}"
    target.write_text(patched)
    if applied > 0:
        _clear_hf_dynamic_module_cache()
        logger.info(
            f"Patched {target} for transformers >= 5.x compatibility "
            f"({applied} applied, {already} already present; HF module cache cleared)."
        )
    return applied > 0
