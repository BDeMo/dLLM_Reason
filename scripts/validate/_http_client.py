"""HTTP client for the dLLM-Reason FastAPI server (scripts/serve.py).

A-axis validation scripts (a3/a4/a5) call the server instead of loading
LLaDA locally. This avoids fighting the server for the GPU and lets one
inference process serve many clients.

Endpoints wrapped:
  /info
  /generate                   (A4 uniform, A5)
  /generate_span_revise       (A3)
  /generate_block_schedule    (A4 non-uniform)
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

# requests is imported lazily inside methods so `--dry_run` works in a minimal
# local env (where only the scope JSON is read, no HTTP is needed).

DEFAULT_URL = os.environ.get("DLLM_SERVER_URL", "http://localhost:8000")


class ValidationAPIClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> dict:
        import requests
        try:
            r = requests.get(f"{self.base_url}/info", timeout=10)
            r.raise_for_status()
            info = r.json()
            if info.get("status") != "ready":
                print(f"[WARN] Server status: {info.get('status')}")
            else:
                print(f"[OK]  server ready: model={info.get('model_id')}  "
                      f"device={info.get('device')}")
            return info
        except requests.ConnectionError:
            print(f"[ERROR] Cannot connect to server at {self.base_url}")
            print("        Start the server first:")
            print("          python scripts/serve.py --model_id <path_or_hf_id>")
            print("        Or set DLLM_SERVER_URL to point to a running instance.")
            sys.exit(1)
        except ModuleNotFoundError:
            print("[ERROR] requests not installed. pip install requests")
            sys.exit(1)

    # ── Generation wrappers ───────────────────────────────────────────────────
    def generate(self, prompt: str, *, strategy: str = "confidence",
                 max_new_tokens: int = 128, num_steps: int = 128,
                 block_length: int = 32, temperature: float = 0.0,
                 remasking: str = "low_confidence") -> str:
        import requests
        r = requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "strategy": strategy,
                "max_new_tokens": max_new_tokens,
                "num_steps": num_steps,
                "block_length": block_length,
                "temperature": temperature,
                "remasking": remasking,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["text"]

    def generate_span_revise(self, prompt: str, *,
                             gen_length: int = 128, steps: int = 128,
                             block_length: int = 32, temperature: float = 0.0,
                             revise_every: int = 8, revise_thresh: float = 0.4,
                             window_size: int = 4) -> str:
        import requests
        r = requests.post(
            f"{self.base_url}/generate_span_revise",
            json={
                "prompt": prompt,
                "gen_length": gen_length,
                "steps": steps,
                "block_length": block_length,
                "temperature": temperature,
                "revise_every": revise_every,
                "revise_thresh": revise_thresh,
                "window_size": window_size,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["text"]

    def generate_block_schedule(self, prompt: str, *,
                                block_sizes: Sequence[int],
                                steps_per_block: Sequence[int],
                                temperature: float = 0.0) -> str:
        import requests
        r = requests.post(
            f"{self.base_url}/generate_block_schedule",
            json={
                "prompt": prompt,
                "block_sizes": list(block_sizes),
                "steps_per_block": list(steps_per_block),
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["text"]

    def generate_inpaint(self, prompt: str, *,
                         anchors: Sequence[tuple[int, str]],
                         gen_length: int = 128, steps: int = 128,
                         block_length: int = 32,
                         temperature: float = 0.0) -> str:
        """Strategy-search inpainting: anchor tokens pre-committed at specified
        positions inside gen region. Used for template_position ∈
        {prefix, suffix, mid, scaffold}."""
        import requests
        r = requests.post(
            f"{self.base_url}/generate_inpaint",
            json={
                "prompt": prompt,
                "anchors": [{"start_pos": p, "text": t} for p, t in anchors],
                "gen_length": gen_length,
                "steps": steps,
                "block_length": block_length,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["text"]


def add_server_arg(ap) -> None:
    """Add --server_url to an argparse parser (default from env DLLM_SERVER_URL)."""
    ap.add_argument("--server_url", type=str, default=DEFAULT_URL,
                    help=f"FastAPI server URL (default: {DEFAULT_URL}; "
                         f"override via --server_url or env DLLM_SERVER_URL)")
