"""
utils/hf_auth.py
────────────────
HuggingFace authentication helper.

Reads HF_TOKEN from the environment (or a .env file if python-dotenv is
installed) and calls huggingface_hub.login() so that gated model downloads
(SD 3.5 Large Turbo, SD 3.5 Medium, etc.) are authorised.

Usage
-----
Call resolve_hf_token() once at process startup before any from_pretrained()
call.  It is safe to call multiple times — login() is idempotent.

    from utils.hf_auth import resolve_hf_token
    resolve_hf_token()          # reads HF_TOKEN from env / .env
    resolve_hf_token("hf_…")   # explicit token (e.g. passed via CLI --hf-token)

Gated models (require accepted license on hf.co + valid token)
--------------------------------------------------------------
    stabilityai/stable-diffusion-3.5-large-turbo
    stabilityai/stable-diffusion-3.5-medium

Non-gated (no token needed)
---------------------------
    runwayml/stable-diffusion-v1-5
    runwayml/stable-diffusion-inpainting
    stabilityai/stable-diffusion-xl-base-0.9
"""

from __future__ import annotations
import os
import warnings

# Gated model prefixes — used to warn early if no token is present
_GATED_PREFIXES = (
    "stabilityai/stable-diffusion-3",
    "stabilityai/stable-diffusion-3.5",
)


def _model_is_gated(model_id: str) -> bool:
    return any(model_id.lower().startswith(p) for p in _GATED_PREFIXES)


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    """
    Resolve and apply a HuggingFace token.

    Priority
    --------
    1. explicit_token argument (e.g. from --hf-token CLI flag)
    2. HF_TOKEN environment variable
    3. HUGGING_FACE_HUB_TOKEN environment variable (legacy name)
    4. .env file (loaded via python-dotenv if available)

    Returns the token string if found, None otherwise.
    Calls huggingface_hub.login() so the token is cached for the session.
    """
    token = explicit_token

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        # Try loading from .env if dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        except ImportError:
            pass

    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("[HF Auth] Authenticated with HuggingFace token.")
        except Exception as e:
            warnings.warn(f"[HF Auth] huggingface_hub.login() failed: {e}", stacklevel=2)
    else:
        print("[HF Auth] No HF_TOKEN found — unauthenticated. Gated models will fail.")

    return token


def check_gated(model_id: str, token: str | None) -> None:
    """
    Emit a clear, actionable warning early if a gated model is requested
    without a token, rather than letting from_pretrained raise an opaque OSError.
    """
    if _model_is_gated(model_id) and not token:
        warnings.warn(
            f"\n[HF Auth] '{model_id}' is a GATED model.\n"
            "  You must:\n"
            "    1. Accept the license at https://huggingface.co/{model_id}\n"
            "    2. Set HF_TOKEN=hf_... in your environment or .env file\n"
            "       (or pass --hf-token hf_... on the CLI)\n"
            "  Without a valid token, from_pretrained() will raise OSError.",
            UserWarning,
            stacklevel=3,
        )
