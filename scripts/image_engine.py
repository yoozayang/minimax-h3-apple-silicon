#!/usr/bin/env python3
"""Unified Local Image Generation Engine on Apple Silicon (MLX / MFLUX)."""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

from PIL import Image

# Dynamic base directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_OUTPUT_DIR = OUTPUTS_DIR / "images"
LOGS_DIR = BASE_DIR / "logs"
IMAGE_HISTORY_FILE = LOGS_DIR / "image_history.jsonl"
ASSETS_FILE = LOGS_DIR / "assets.jsonl"

IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Import model manager
try:
    from scripts import model_manager
except ImportError:
    import model_manager


@dataclass
class ImageResult:
    id: str
    asset_id: str
    success: bool
    output_path: str
    output_filename: str
    prompt: str
    seed: int
    width: int
    height: int
    steps: int
    model_name: str
    execution_time_sec: float
    created_at: str
    error_message: str | None = None


# Cache for the loaded flux instance
_RESIDENT_FLUX_MODEL: Any = None
_RESIDENT_MODEL_NAME: str | None = None


def _unload_image_model():
    global _RESIDENT_FLUX_MODEL, _RESIDENT_MODEL_NAME
    _RESIDENT_FLUX_MODEL = None
    _RESIDENT_MODEL_NAME = None


# Register with model manager
model_manager.register_unload_callback("IMAGE", _unload_image_model)


def register_asset(
    asset_type: str,
    source: str,
    file_path: str | Path,
    metadata: dict | None = None,
    prompt: str | None = None,
) -> dict:
    """Register a media file into the unified asset library."""
    p = Path(file_path).expanduser().resolve()
    asset_id = "ast_" + str(uuid.uuid4())[:8]
    record = {
        "id": asset_id,
        "type": asset_type.upper(),  # "IMAGE", "VIDEO"
        "source": source.upper(),    # "UPLOAD", "GENERATED", "REFERENCE"
        "path": str(p),
        "filename": p.name,
        "created_at": datetime.now().isoformat(),
        "prompt": prompt or "",
        "metadata": metadata or {},
    }
    try:
        with open(ASSETS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to save asset record: {e}", file=sys.stderr)
    return record


def get_image_history(limit: int = 40) -> list[dict]:
    """Retrieve recent image generation records."""
    if not IMAGE_HISTORY_FILE.exists():
        return []
    records = []
    try:
        with open(IMAGE_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        p = item.get("output_path")
                        item["file_exists"] = bool(p and Path(p).exists() and Path(p).is_file())
                        records.append(item)
                    except json.JSONDecodeError:
                        continue
        return records[::-1][:limit]
    except Exception as e:
        print(f"Warning: Failed to read image history: {e}", file=sys.stderr)
        return []


def generate_images(
    prompt: str,
    width: int = 768,
    height: int = 768,
    steps: int = 4,
    seed: int = -1,
    model_name: str = "schnell",
    quantize: int = 4,
    count: int = 1,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[ImageResult]:
    """Generate 1 to 4 images sequentially using MFLUX on Apple Silicon."""
    global _RESIDENT_FLUX_MODEL, _RESIDENT_MODEL_NAME

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Step 1: Switch engine residency
    model_manager.switch_to_engine("IMAGE")

    # Align dimensions to multiples of 16
    width = (width // 16) * 16
    height = (height // 16) * 16
    width = max(256, min(1536, width))
    height = max(256, min(1536, height))

    results: list[ImageResult] = []
    count = max(1, min(4, count))

    if progress_callback:
        progress_callback(0.05, f"正在載入 {model_name.upper()} (MLX {quantize}-bit) 圖像模型...")

    # Load model if not resident
    model_key = f"{model_name}_{quantize}bit"
    if _RESIDENT_FLUX_MODEL is None or _RESIDENT_MODEL_NAME != model_key:
        try:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            _RESIDENT_FLUX_MODEL = Flux1.from_name(model_name=model_name, quantize=quantize)
            _RESIDENT_MODEL_NAME = model_key
        except Exception as e:
            print(f"[ImageEngine] Error loading MFLUX model: {e}", file=sys.stderr)
            raise RuntimeError(f"無法載入 MFLUX 模型: {e}")

    for idx in range(count):
        if cancel_check and cancel_check():
            break

        current_seed = (seed + idx) if seed >= 0 else random.randint(0, 2**31 - 1)
        start_time = time.time()

        if progress_callback:
            progress_callback(
                0.1 + 0.85 * (idx / count),
                f"正在生成第 {idx+1}/{count} 張圖片 (Seed: {current_seed})...",
            )

        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp_str}_img_seed{current_seed}_{width}x{height}.png"
        out_path = IMAGES_OUTPUT_DIR / filename

        try:
            generated_img = _RESIDENT_FLUX_MODEL.generate_image(
                seed=current_seed,
                prompt=prompt.strip(),
                num_inference_steps=steps,
                height=height,
                width=width,
            )

            # Save PIL image
            generated_img.image.save(str(out_path), "PNG")
            exec_time = time.time() - start_time

            # Register into asset library
            asset_record = register_asset(
                asset_type="IMAGE",
                source="GENERATED",
                file_path=out_path,
                prompt=prompt.strip(),
                metadata={"seed": current_seed, "width": width, "height": height, "steps": steps, "model": model_key},
            )

            res = ImageResult(
                id=str(uuid.uuid4())[:8],
                asset_id=asset_record["id"],
                success=True,
                output_path=str(out_path),
                output_filename=filename,
                prompt=prompt.strip(),
                seed=current_seed,
                width=width,
                height=height,
                steps=steps,
                model_name=model_key,
                execution_time_sec=round(exec_time, 2),
                created_at=datetime.now().strftime("%H:%M:%S"),
            )

            # Append to history JSONL
            try:
                with open(IMAGE_HISTORY_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write image history: {e}", file=sys.stderr)

            results.append(res)

        except Exception as e:
            exec_time = time.time() - start_time
            print(f"[ImageEngine] Error generating image #{idx+1}: {e}", file=sys.stderr)
            res = ImageResult(
                id=str(uuid.uuid4())[:8],
                asset_id="",
                success=False,
                output_path="",
                output_filename="",
                prompt=prompt.strip(),
                seed=current_seed,
                width=width,
                height=height,
                steps=steps,
                model_name=model_key,
                execution_time_sec=round(exec_time, 2),
                created_at=datetime.now().strftime("%H:%M:%S"),
                error_message=str(e),
            )
            results.append(res)

    if progress_callback:
        progress_callback(1.0, f"圖片生成完成！已儲存 {len([r for r in results if r.success])} 張圖片。")

    return results
