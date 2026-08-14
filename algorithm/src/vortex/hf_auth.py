"""HuggingFace Hub auth for the gated Llama-2 / Mistral checkpoints these
pipelines download. Reads the token from `~/vortex/hf_token` (a plain-text
file containing just the token, one line) rather than requiring
`huggingface-cli login` to have been run interactively -- needed because the
pipelines are launched unattended, inside tmux.
"""
import logging

from .paths import HF_TOKEN_PATH

_logged_in = False


def read_hf_token() -> str:
    if not HF_TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"{HF_TOKEN_PATH} not found. Create it with a single line "
            f"containing your HuggingFace access token (needs read access "
            f"to meta-llama/* and mistralai/* gated repos)."
        )
    token = HF_TOKEN_PATH.read_text().strip()
    if not token:
        raise ValueError(f"{HF_TOKEN_PATH} is empty.")
    return token


def ensure_hf_login():
    """Idempotent: logs in once per process via huggingface_hub, so every
    from_pretrained() call in this process picks up the token without every
    call site having to pass token= explicitly."""
    global _logged_in
    if _logged_in:
        return
    token = read_hf_token()
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    logging.info(f"Logged in to HuggingFace Hub using token from {HF_TOKEN_PATH}")
    _logged_in = True
