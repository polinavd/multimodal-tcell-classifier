"""
Automatic model weight download and caching.

On first run, downloads ensemble weights and encoders from Hugging Face Hub
(or GitHub Releases as fallback) to ~/.cache/tcell-classifier/.
Subsequent runs use the cached files.

Supports:
  - huggingface_hub (preferred, pip install huggingface-hub)
  - Direct HTTPS download (fallback, no extra dependencies)
"""

import os
import sys
import hashlib
from pathlib import Path

# Default remote locations
HF_REPO_ID = "VirialyD/tcell-classifier"
GITHUB_RELEASE_URL = (
    "https://github.com/polinavd/multimodal-tcell-classifier/releases/download/v1.0"
)

# Files required for inference
REQUIRED_FILES = {
    "weights": [
        "m1_h512.pt",
        "m2_h512_s2.pt",
        "m4_8heads.pt",
        "m6_highdrop.pt",
        "m7_lr3e4.pt",
        "results.json",
    ],
    "encoders": [
        "label_encoder.pkl",
        "vj_encoder.pkl",
    ],
}

CACHE_DIR_ENV = "TCELL_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tcell-classifier"


def get_cache_dir() -> Path:
    """Get cache directory, respecting TCELL_CACHE_DIR env var."""
    cache = Path(os.environ.get(CACHE_DIR_ENV, DEFAULT_CACHE_DIR))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def get_model_dir() -> Path:
    """Get path to cached model weights. Downloads if missing."""
    return ensure_weights()


def _check_cached(cache_dir: Path) -> bool:
    """Check if all required files exist in cache."""
    for group in REQUIRED_FILES.values():
        for fname in group:
            if not (cache_dir / fname).exists():
                return False
    return True


def _download_from_hf(cache_dir: Path) -> bool:
    """Try downloading from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download

        all_files = []
        for group in REQUIRED_FILES.values():
            all_files.extend(group)

        print(f"Downloading model files from Hugging Face ({HF_REPO_ID})...")
        for i, fname in enumerate(all_files, 1):
            print(f"  [{i}/{len(all_files)}] {fname}")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  Hugging Face download failed: {e}")
        return False


def _download_from_github(cache_dir: Path) -> bool:
    """Fallback: download from GitHub Releases via urllib."""
    try:
        from urllib.request import urlretrieve

        all_files = []
        for group in REQUIRED_FILES.values():
            all_files.extend(group)

        print(f"Downloading model files from GitHub Releases...")
        for i, fname in enumerate(all_files, 1):
            url = f"{GITHUB_RELEASE_URL}/{fname}"
            dest = cache_dir / fname
            print(f"  [{i}/{len(all_files)}] {fname}")
            urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  GitHub download failed: {e}")
        return False


def ensure_weights(cache_dir: Path | None = None) -> Path:
    """
    Ensure model weights are available locally.

    Checks cache first, then downloads from HF Hub or GitHub Releases.

    Args:
        cache_dir: Override cache directory (default: ~/.cache/tcell-classifier)

    Returns:
        Path to directory containing all model files

    Raises:
        RuntimeError: If download fails from all sources
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    if _check_cached(cache_dir):
        return cache_dir

    print(f"\nFirst run — downloading model weights to {cache_dir}")
    print("This only happens once (~500 MB).\n")

    # Try Hugging Face first, then GitHub
    if _download_from_hf(cache_dir):
        if _check_cached(cache_dir):
            print("\nDownload complete.\n")
            return cache_dir

    if _download_from_github(cache_dir):
        if _check_cached(cache_dir):
            print("\nDownload complete.\n")
            return cache_dir

    raise RuntimeError(
        "Failed to download model weights.\n\n"
        "Options:\n"
        f"  1. Install huggingface-hub: pip install huggingface-hub\n"
        f"  2. Download manually from {GITHUB_RELEASE_URL}\n"
        f"  3. Place files in {cache_dir}/\n\n"
        f"Required files: {[f for g in REQUIRED_FILES.values() for f in g]}"
    )
