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


# Registry of available image generation models
IMAGE_MODELS: dict[str, dict[str, Any]] = {
    "fhdr-uncensored": {
        "id": "fhdr-uncensored",
        "display_name": "FHDR Uncensored — Quality",
        "repo_id": "PocketAiHub/FHDR-Uncensored-MFLUX",
        "backend": "mlx_mflux",
        "supports_t2i": True,
        "supports_i2i": True,
        "supports_multi_reference": False,
        "supported_quantization": [4, 8],
        "recommended_profiles": ["draft", "balanced", "high", "maximum"],
        "default_profile": "high",
        "memory_requirement": "~10-14 GB (Q4) / ~18-22 GB (Q8)",
        "license": "FLUX.1-dev Non-Commercial License Derivative",
        "is_default": True,
        "description": "高品質無閹割寫實模型，創作自由度高、真實光影與細節豐富",
        "is_gated": True,
        "available": True,
    },
    "flux2-klein-4b": {
        "id": "flux2-klein-4b",
        "display_name": "FLUX.2 Klein 4B — Fast",
        "repo_id": "black-forest-labs/FLUX.2-Klein-4B",
        "backend": "mlx_mflux",
        "supports_t2i": True,
        "supports_i2i": True,
        "supports_multi_reference": False,
        "supported_quantization": [4, 8],
        "recommended_profiles": ["draft", "balanced", "high", "maximum"],
        "default_profile": "high",
        "memory_requirement": "~3-5 GB",
        "license": "Apache 2.0 / Open Weights",
        "is_default": False,
        "description": "極速備用模型，4步快速構圖，開源免授權",
        "is_gated": False,
        "available": True,
    },
}


def check_hf_token_available() -> bool:
    """Check if Hugging Face token is present in env or local cache."""
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    token_path_alt = Path.home() / ".huggingface" / "token"
    return bool(
        os.environ.get("HF_TOKEN")
        or (token_path.exists() and token_path.read_text().strip())
        or (token_path_alt.exists() and token_path_alt.read_text().strip())
    )


def save_hf_token(token: str) -> None:
    """Save Hugging Face token to local user cache."""
    t = token.strip()
    if not t:
        raise ValueError("Token cannot be empty")
    token_dir = Path.home() / ".cache" / "huggingface"
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / "token").write_text(t, encoding="utf-8")
    os.environ["HF_TOKEN"] = t


def get_available_image_models() -> dict[str, Any]:
    """Retrieve available image models, capabilities, and default selection."""
    has_token = check_hf_token_available()
    models = []
    for m in IMAGE_MODELS.values():
        item = dict(m)
        if item.get("is_gated") and not has_token:
            item["auth_status"] = "needs_token"
        else:
            item["auth_status"] = "ready"
        models.append(item)
    return {
        "default_model": "fhdr-uncensored",
        "has_hf_token": has_token,
        "models": models,
    }


def _calculate_resolution_by_class(target_class: int, width: int = 768, height: int = 768) -> tuple[int, int]:
    """Scale requested aspect ratio (width:height) to match target pixel budget (target_class^2), aligned to 16."""
    w = max(256, width)
    h = max(256, height)
    aspect = w / h
    target_area = float(target_class * target_class)

    new_h = (target_area / aspect) ** 0.5
    new_w = new_h * aspect

    aligned_w = max(256, min(1536, int(round(new_w / 16.0)) * 16))
    aligned_h = max(256, min(1536, int(round(new_h / 16.0)) * 16))
    return aligned_w, aligned_h


@dataclass
class ResolvedImageConfig:
    model_name: str
    steps: int
    quantize: int
    width: int
    height: int
    guidance: float
    scheduler: str | None = None
    quality_profile: str = "high"


def resolve_image_profile(
    model_name: str = "fhdr-uncensored",
    quality_profile: str = "high",
    width: int = 768,
    height: int = 768,
    custom_steps: int | None = None,
    custom_quantize: int | None = None,
) -> ResolvedImageConfig:
    """Resolve model-specific inference parameters for Draft/Balanced/High/Maximum preserving aspect ratio."""
    model_key = "flux2-klein-4b" if "klein" in str(model_name).lower() else "fhdr-uncensored"
    qp = (quality_profile or "high").lower()

    if qp == "custom":
        w = max(256, min(1536, (width // 16) * 16))
        h = max(256, min(1536, (height // 16) * 16))
        return ResolvedImageConfig(
            model_name=model_key,
            steps=custom_steps if custom_steps and custom_steps > 0 else (40 if model_key == "fhdr-uncensored" else 4),
            quantize=custom_quantize if custom_quantize in [4, 8] else 4,
            width=w,
            height=h,
            guidance=4.0 if model_key == "fhdr-uncensored" else 1.0,
            quality_profile="custom",
        )

    if model_key == "fhdr-uncensored":
        if qp == "draft":
            # Draft: Q4 / 12 steps / 512-class / Guidance 4.0
            w, h = _calculate_resolution_by_class(512, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=12,
                quantize=4,
                width=w,
                height=h,
                guidance=4.0,
                quality_profile="draft",
            )
        elif qp == "balanced":
            # Balanced: Q4 / 24 steps / 768-class / Guidance 4.0
            w, h = _calculate_resolution_by_class(768, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=24,
                quantize=4,
                width=w,
                height=h,
                guidance=4.0,
                quality_profile="balanced",
            )
        elif qp == "maximum":
            # Maximum: 1024-class, Guidance 4.0 (to be finalized based on benchmark between Q8/40, Q4/48, Q4/Higher Res)
            w, h = _calculate_resolution_by_class(1024, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=40,
                quantize=8,
                width=w,
                height=h,
                guidance=4.0,
                quality_profile="maximum",
            )
        else:  # "high" (Default)
            # High: Q4 / 40 steps / 1024-class / Guidance 4.0
            w, h = _calculate_resolution_by_class(1024, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=40,
                quantize=4,
                width=w,
                height=h,
                guidance=4.0,
                quality_profile="high",
            )
    else:  # FLUX.2 Klein 4B
        if qp == "draft":
            w, h = _calculate_resolution_by_class(512, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=2,
                quantize=4,
                width=w,
                height=h,
                guidance=1.0,
                quality_profile="draft",
            )
        elif qp == "balanced":
            w, h = _calculate_resolution_by_class(768, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=4,
                quantize=4,
                width=w,
                height=h,
                guidance=1.0,
                quality_profile="balanced",
            )
        elif qp == "maximum":
            w, h = _calculate_resolution_by_class(1024, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=8,
                quantize=4,
                width=w,
                height=h,
                guidance=1.0,
                quality_profile="maximum",
            )
        else:  # "high" (Default)
            w, h = _calculate_resolution_by_class(1024, width, height)
            return ResolvedImageConfig(
                model_name=model_key,
                steps=4,
                quantize=4,
                width=w,
                height=h,
                guidance=1.0,
                quality_profile="high",
            )


def save_image_history_record(res: ImageResult) -> None:
    """Append image generation result to history JSONL."""
    try:
        with open(IMAGE_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to save image history record: {e}", file=sys.stderr)


def generate_images(
    prompt: str,
    width: int = 768,
    height: int = 768,
    steps: int = 4,
    seed: int = -1,
    model_name: str = "fhdr-uncensored",
    quality_profile: str = "high",
    quantize: int = 4,
    count: int = 1,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[ImageResult]:
    """Generate 1 to 4 images sequentially using FHDR Uncensored / FLUX.2 Klein on Apple Silicon."""
    global _RESIDENT_FLUX_MODEL, _RESIDENT_MODEL_NAME

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Step 1: Switch engine residency
    model_manager.switch_to_engine("IMAGE")

    # Step 2: Resolve quality profile
    cfg = resolve_image_profile(
        model_name=model_name,
        quality_profile=quality_profile,
        width=width,
        height=height,
        custom_steps=steps,
        custom_quantize=quantize,
    )

    eff_width = cfg.width
    eff_height = cfg.height
    eff_steps = cfg.steps
    eff_quantize = cfg.quantize
    eff_guidance = cfg.guidance
    eff_model = cfg.model_name
    eff_profile = cfg.quality_profile

    results: list[ImageResult] = []
    count = max(1, min(4, count))

    # Resolve output directory
    target_dir = Path(output_dir).expanduser().resolve() if output_dir and str(output_dir).strip() else IMAGES_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.05, f"正在載入 {eff_model.upper()} ({eff_profile.upper()} · {eff_quantize}-bit) 模型...")

    # Load model if not resident
    model_key = f"{eff_model}_{eff_quantize}bit"
    if _RESIDENT_FLUX_MODEL is None or _RESIDENT_MODEL_NAME != model_key:
        try:
            # Cleanly unload prior model and purge Metal cache
            if _RESIDENT_FLUX_MODEL is not None:
                del _RESIDENT_FLUX_MODEL
                _RESIDENT_FLUX_MODEL = None
                gc.collect()
                try:
                    import mlx.core as mx
                    mx.metal.clear_cache()
                except Exception:
                    pass

            if eff_model == "fhdr-uncensored":
                # Verify Hugging Face access for gated repo
                if not check_hf_token_available():
                    raise RuntimeError(
                        "載入 PocketAiHub/FHDR-Uncensored-MFLUX 需要 Hugging Face Access Token 授權。\n"
                        "請至 https://huggingface.co/settings/tokens 取得 Token 並在設定中填入，或切換至 FLUX.2 Klein 4B 繼續生成。"
                    )
                from mflux.models.flux.variants.txt2img.flux import Flux1
                from mflux.models.common.config.model_config import ModelConfig
                _RESIDENT_FLUX_MODEL = Flux1(
                    model_path="PocketAiHub/FHDR-Uncensored-MFLUX",
                    quantize=eff_quantize,
                    model_config=ModelConfig.dev(),
                )
            else:  # flux2-klein-4b
                from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
                from mflux.models.common.config.model_config import ModelConfig
                flux_cfg = ModelConfig.flux2_klein_4b()
                _RESIDENT_FLUX_MODEL = Flux2Klein(quantize=eff_quantize, model_config=flux_cfg)

            _RESIDENT_MODEL_NAME = model_key
        except Exception as e:
            print(f"[ImageEngine] Error loading model ({eff_model}): {e}", file=sys.stderr)
            raise RuntimeError(f"無法載入模型 ({eff_model}): {e}")

    for idx in range(count):
        if cancel_check and cancel_check():
            break

        current_seed = (seed + idx) if seed >= 0 else random.randint(0, 2**31 - 1)
        start_time = time.time()

        if progress_callback:
            progress_callback(
                0.1 + 0.85 * (idx / count),
                f"正在生成第 {idx+1}/{count} 張圖片 ({eff_model.upper()} · {eff_profile} · Seed: {current_seed})...",
            )

        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp_str}_{eff_model}_seed{current_seed}_{eff_width}x{eff_height}.png"
        out_path = target_dir / filename

        try:
            if eff_model == "fhdr-uncensored":
                generated_img = _RESIDENT_FLUX_MODEL.generate_image(
                    seed=current_seed,
                    prompt=prompt.strip(),
                    num_inference_steps=eff_steps,
                    height=eff_height,
                    width=eff_width,
                    guidance=eff_guidance,
                )
            else:
                generated_img = _RESIDENT_FLUX_MODEL.generate_image(
                    seed=current_seed,
                    prompt=prompt.strip(),
                    num_inference_steps=eff_steps,
                    height=eff_height,
                    width=eff_width,
                )

            # Save PIL image
            generated_img.image.save(str(out_path), "PNG")
            exec_time = time.time() - start_time

            # Register into asset library
            asset_record = register_asset(
                asset_type="IMAGE",
                source="GENERATED",
                file_path=out_path,
                metadata={
                    "width": eff_width,
                    "height": eff_height,
                    "steps": eff_steps,
                    "seed": current_seed,
                    "model": eff_model,
                    "quality_profile": eff_profile,
                    "quantize": eff_quantize,
                    "guidance": eff_guidance,
                },
                prompt=prompt.strip(),
            )

            res = ImageResult(
                id=str(uuid.uuid4())[:8],
                asset_id=asset_record["id"],
                success=True,
                output_path=str(out_path),
                output_filename=filename,
                prompt=prompt.strip(),
                seed=current_seed,
                width=eff_width,
                height=eff_height,
                steps=eff_steps,
                model_name=eff_model,
                execution_time_sec=round(exec_time, 2),
                created_at=datetime.now().isoformat(),
            )
            results.append(res)
            save_image_history_record(res)

        except Exception as e:
            print(f"[ImageEngine] Error generating image: {e}", file=sys.stderr)
            res = ImageResult(
                id=str(uuid.uuid4())[:8],
                asset_id="",
                success=False,
                output_path="",
                output_filename="",
                prompt=prompt.strip(),
                seed=current_seed,
                width=eff_width,
                height=eff_height,
                steps=eff_steps,
                model_name=eff_model,
                execution_time_sec=round(time.time() - start_time, 2),
                created_at=datetime.now().isoformat(),
                error_message=str(e),
            )
            results.append(res)
            save_image_history_record(res)
            raise e

    return results
