#!/usr/bin/env python3
"""High-speed model download script for MiniMax-H3 8-bit MLX."""

import os
import sys
import time
from pathlib import Path

# Enable Rust-based hf_transfer for maximum download throughput
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download

REPO_ID = "appautomaton/minimax-h3-base-8bit-mlx"
MODELS_DIR = Path("~/AI/minimax-h3/models").expanduser()

REQUIRED_FILES = [
    "tokenizer/tokenizer.json",
    "bf16/vae/minimax_h3_audio_vae_fp32.safetensors",
    "bf16/vae/minimax_h3_video_vae_fp16.safetensors",
    "mlx-8bit/te_qwen3vl_a8g32.safetensors",
    "mlx-8bit/dit_fl2va_a8g32.safetensors",
]

def download_all():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== MiniMax-H3 MLX 8-bit Model Downloader (hf-transfer enabled) ===", flush=True)
    print(f"Repository: {REPO_ID}", flush=True)
    print(f"Target Directory: {MODELS_DIR}", flush=True)
    print(f"Required Files: {len(REQUIRED_FILES)}", flush=True)
    print("-" * 50, flush=True)

    total_start = time.time()
    for idx, filename in enumerate(REQUIRED_FILES, 1):
        target_path = MODELS_DIR / filename
        if target_path.exists() and target_path.stat().st_size > 1000:
            size_gb = target_path.stat().st_size / (1024**3)
            print(f"[{idx}/{len(REQUIRED_FILES)}] Already exists ({size_gb:.2f} GB): {filename}", flush=True)
            continue

        print(f"\n[{idx}/{len(REQUIRED_FILES)}] Downloading {filename} ...", flush=True)
        file_start = time.time()
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            file_size = Path(path).stat().st_size / (1024**3)
            elapsed = time.time() - file_start
            speed_mb = (file_size * 1024) / max(elapsed, 0.1)
            print(f"✓ Completed {filename} ({file_size:.2f} GB in {elapsed:.1f}s, {speed_mb:.1f} MB/s)", flush=True)
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    print("\n" + "=" * 50, flush=True)
    print(f"All required models downloaded successfully in {(time.time() - total_start)/60:.1f} min!", flush=True)
    print("=" * 50, flush=True)

if __name__ == "__main__":
    download_all()
