#!/usr/bin/env python3
"""Unified generation backend for MiniMax-H3 on Apple Silicon."""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# Dynamically resolve BASE_DIR to project root (e.g. /Users/yoozayang/Minimax)
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_SRC = BASE_DIR / "repo" / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as np
import mlx.core as mx
from mlx_h3 import layout, memory, output, pipeline, sampler

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
HISTORY_FILE = LOGS_DIR / "history.jsonl"


@dataclass
class GenerationResult:
    success: bool
    output_path: str
    output_filename: str
    output_dir: str
    prompt: str
    seed: int
    width: int
    height: int
    frames: int
    duration_sec: float
    steps: int
    execution_time_sec: float
    peak_memory_gib: float
    subtitle_text: str | None = None
    error_message: str | None = None


PROFILES = {
    "fast": {
        "name": "快速測試",
        "description": "768×448 / ~2s (16 frames, 10 steps)",
        "width": 768,
        "height": 448,
        "duration_sec": 2.0,
        "frames": 16,
        "steps": 10,
    },
    "standard": {
        "name": "標準 540p",
        "description": "960×544 / ~2.5s (20 frames, 15 steps)",
        "width": 960,
        "height": 544,
        "duration_sec": 2.5,
        "frames": 20,
        "steps": 15,
    },
    "720p_short": {
        "name": "720p Short",
        "description": "1280×720 / ~2s (16 frames, 15 steps)",
        "width": 1280,
        "height": 720,
        "duration_sec": 2.0,
        "frames": 16,
        "steps": 15,
    },
    "720p": {
        "name": "720p 標準",
        "description": "1280×720 / ~3s (24 frames, 20 steps)",
        "width": 1280,
        "height": 720,
        "duration_sec": 3.0,
        "frames": 24,
        "steps": 20,
    },
}


def get_system_memory_status() -> dict:
    """Retrieve system memory pressure and swap usage."""
    try:
        sample = memory.sample()
        device_info = mx.device_info()
        max_working_gib = device_info.get("max_recommended_working_set_size", 24 * (1 << 30)) / (1024**3)
        total_ram_gib = device_info.get("memory_size", 32 * (1 << 30)) / (1024**3)
        swap_used_mb = sample.swap_used / (1024**2)

        # Free percentage level from kern.memorystatus_level
        pressure = "Normal"
        if sample.free_pct < 20 or swap_used_mb > 2048:
            pressure = "Warning"
        elif sample.free_pct < 10 or swap_used_mb > 5120:
            pressure = "Critical"

        return {
            "pressure": pressure,
            "free_pct": sample.free_pct,
            "swap_used_mb": round(swap_used_mb, 1),
            "swap_used_gib": round(swap_used_mb / 1024, 2),
            "active_gib": round(sample.active / (1024**3), 2),
            "peak_gib": round(sample.peak / (1024**3), 2),
            "total_ram_gib": round(total_ram_gib, 1),
            "max_working_gib": round(max_working_gib, 1),
        }
    except Exception as e:
        return {
            "pressure": "Unknown",
            "free_pct": 50,
            "swap_used_mb": 0.0,
            "swap_used_gib": 0.0,
            "active_gib": 0.0,
            "peak_gib": 0.0,
            "total_ram_gib": 32.0,
            "max_working_gib": 25.0,
            "error": str(e),
        }


def align_canvas(width: int, height: int) -> tuple[int, int]:
    """Align width and height to layout.CANVAS_MULTIPLE (16)."""
    w = max(32, (width // layout.CANVAS_MULTIPLE) * layout.CANVAS_MULTIPLE)
    h = max(32, (height // layout.CANVAS_MULTIPLE) * layout.CANVAS_MULTIPLE)
    if w * h > layout.MAX_PIXELS:
        scale = (layout.MAX_PIXELS / (w * h)) ** 0.5
        w = max(32, int(w * scale // 16) * 16)
        h = max(32, int(h * scale // 16) * 16)
    return w, h


def resolve_frames(duration_sec: float, fps: int = layout.FPS) -> int:
    """Convert duration to valid frame count."""
    raw_frames = max(5, int(duration_sec * fps))
    aligned = layout.align_frame_count(raw_frames)
    return min(aligned, 362)


def get_model_paths() -> pipeline.ModelPaths:
    """Return model paths anchored at ~/AI/minimax-h3/models."""
    return pipeline.ModelPaths(
        tokenizer=MODELS_DIR / "tokenizer" / "tokenizer.json",
        text_encoder=MODELS_DIR / "mlx-8bit" / "te_qwen3vl_a8g32.safetensors",
        dit=MODELS_DIR / "mlx-8bit" / "dit_fl2va_a8g32.safetensors",
        ref_dit=MODELS_DIR / "mlx-8bit" / "dit_ref2va_a8g32.safetensors",
        video_vae=MODELS_DIR / "bf16" / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        audio_vae=MODELS_DIR / "bf16" / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
    )


def append_history(result: GenerationResult):
    """Append execution result to logs/history.jsonl."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            **asdict(result),
        }
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to append history: {e}", file=sys.stderr)


def get_history(limit: int = 20) -> list[dict]:
    """Retrieve history of generations."""
    if not HISTORY_FILE.exists():
        return []
    records = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        # Fix path if file is located in workspace outputs
                        if item.get("output_filename"):
                            candidate = OUTPUTS_DIR / item["output_filename"]
                            if candidate.exists():
                                item["output_path"] = str(candidate)
                        records.append(item)
                    except json.JSONDecodeError:
                        continue
        return records[::-1][:limit]
    except Exception as e:
        print(f"Warning: Failed to read history: {e}", file=sys.stderr)
        return []


def write_srt_file(srt_path: str | Path, text: str, duration_sec: float) -> None:
    """Generate standard UTF-8 SRT subtitle file."""
    if not text or not text.strip():
        return
    end_h = int(duration_sec // 3600)
    end_m = int((duration_sec % 3600) // 60)
    end_s = int(duration_sec % 60)
    end_ms = int((duration_sec * 1000) % 1000)
    time_code = f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"

    content = f"1\n00:00:00,000 --> {time_code}\n{text.strip()}\n"
    try:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Warning: Failed to write SRT file: {e}", file=sys.stderr)


def render_subtitles_on_frames(
    frames: np.ndarray,
    text: str,
    position: str = "bottom",
    style: str = "box",
    font_size: int = 24,
) -> np.ndarray:
    """Render subtitle text onto each frame using PIL with anti-aliasing."""
    if not text or not text.strip():
        return frames

    try:
        from PIL import Image, ImageDraw, ImageFont

        # Try to load Apple system font (PingFang TC / Helvetica / STHeiti)
        font = None
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for fp in font_paths:
            if Path(fp).exists():
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        # Determine styles
        text_color = (255, 255, 255)
        stroke_color = (0, 0, 0)
        stroke_w = max(1, font_size // 10)
        box_bg = (0, 0, 0, 160)

        if style == "yellow":
            text_color = (255, 235, 59)
            stroke_color = (0, 0, 0)
        elif style == "stroke":
            box_bg = None

        rendered_frames = []
        clean_text = text.strip()

        # Process each frame
        for frame in frames:
            img = Image.fromarray(frame).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # Auto text-wrapping for wide lines
            max_w = int(img.width * 0.90)
            raw_lines = clean_text.split("\n")
            lines = []
            for raw_l in raw_lines:
                line = ""
                for ch in raw_l:
                    test_line = line + ch
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    if bbox[2] - bbox[0] > max_w and line:
                        lines.append(line)
                        line = ch
                    else:
                        line = test_line
                if line:
                    lines.append(line)

            line_height = int(font_size * 1.3)
            total_text_h = len(lines) * line_height

            if position == "top":
                start_y = int(img.height * 0.08)
            else:  # bottom
                start_y = img.height - total_text_h - int(img.height * 0.08)

            for i, line_str in enumerate(lines):
                bbox = draw.textbbox((0, 0), line_str, font=font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                lx = (img.width - lw) // 2
                ly = start_y + i * line_height

                if box_bg:
                    pad = int(font_size * 0.35)
                    draw.rounded_rectangle(
                        [lx - pad, ly - pad // 2, lx + lw + pad, ly + lh + pad // 2],
                        radius=6,
                        fill=box_bg,
                    )
                draw.text(
                    (lx, ly),
                    line_str,
                    font=font,
                    fill=text_color + (255,),
                    stroke_width=stroke_w,
                    stroke_fill=stroke_color + (255,),
                )

            final_img = Image.alpha_composite(img, overlay).convert("RGB")
            rendered_frames.append(np.array(final_img))

        return np.stack(rendered_frames)
    except Exception as e:
        print(f"Subtitle rendering warning (fallback to clean frames): {e}", flush=True)
        return frames


def generate_video(
    prompt: str,
    width: int = 768,
    height: int = 448,
    duration_sec: float = 2.0,
    frames: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    on_stage: Callable[[str, float], None] | None = None,
    output_dir: str | Path | None = None,
    subtitle_text: str | None = None,
    subtitle_position: str = "bottom",
    subtitle_style: str = "box",
    subtitle_font_size: int = 24,
    first_frame: str | Path | None = None,
    last_frame: str | Path | None = None,
) -> GenerationResult:
    """Core video and audio generation backend.

    Called by both CLI and Web UI.
    """
    # Ensure target output directory is valid and writable (fallback to project outputs if macOS permission fails)
    target_out_dir = Path(output_dir).expanduser().resolve() if output_dir else OUTPUTS_DIR
    try:
        target_out_dir.mkdir(parents=True, exist_ok=True)
        test_file = target_out_dir / ".minimax_perm_test"
        test_file.touch()
        test_file.unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: Custom output dir '{target_out_dir}' not writable ({e}), falling back to project outputs.", file=sys.stderr)
        target_out_dir = OUTPUTS_DIR
        target_out_dir.mkdir(parents=True, exist_ok=True)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # Clean previous memory footprint
    gc.collect()
    memory.release()

    # Resolve seed
    actual_seed = seed if (seed is not None and seed >= 0) else random.randint(1, 2147483647)

    # Resolve canvas and frames
    actual_width, actual_height = align_canvas(width, height)
    actual_frames = frames if frames is not None else resolve_frames(duration_sec)
    actual_duration = round(actual_frames / layout.FPS, 2)

    # Model paths
    paths = get_model_paths()
    actual_steps = paths.sampling_profile.resolve_steps(steps)

    def report_stage(stage_text: str, progress: float):
        if on_stage:
            on_stage(stage_text, progress)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {stage_text}", flush=True)

    report_stage("Checking models and system memory...", 0.02)
    paths.validate(ref2va=False)

    memory.configure(budget_gib=28)
    guard = memory.Guard("generate", budget_gib=28)

    started_time = time.perf_counter()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_filename = f"{time_str}_seed{actual_seed}_{actual_width}x{actual_height}.mp4"
    out_path = str(target_out_dir / out_filename)
    log_filename = f"{time_str}_seed{actual_seed}.log"
    log_path = str(LOGS_DIR / log_filename)

    config = pipeline.GenerationConfig(
        prompt=prompt.strip(),
        width=actual_width,
        height=actual_height,
        frames=actual_frames,
        seed=actual_seed,
        steps=actual_steps,
        first_frame=first_frame,
        last_frame=last_frame,
    )

    peak_memory_bytes = 0

    def on_phase_report(item: pipeline.PhaseReport):
        nonlocal peak_memory_bytes
        if item.peak > peak_memory_bytes:
            peak_memory_bytes = item.peak

        if "text" in item.label.lower():
            report_stage(
                f"[2/4] 提示詞編碼完成 (Peak {item.peak/(1024**3):.1f} GB)！正在載入 33B DiT 擴散模型...",
                0.15,
            )
        elif "dit" in item.label.lower():
            report_stage(
                f"[3/4] 擴散去噪全部完成 (Peak {item.peak/(1024**3):.1f} GB)！正在使用 Video VAE 進行視訊解碼...",
                0.80,
            )
        elif "video" in item.label.lower():
            report_stage(
                f"[4/4] 視訊解碼完成！正在進行音訊解碼與字幕封裝...",
                0.90,
            )
        else:
            report_stage(
                f"Completed: {item.label} (Peak: {item.peak/(1024**3):.1f} GB)",
                0.92,
            )

    def on_step_progress(done: int, total: int, sigma_video: float, sigma_audio: float):
        pct = done / total
        progress = 0.15 + pct * 0.65
        est_min_left = max(0, round((total - done) * 1.9, 1))
        report_stage(
            f"[2/4] 擴散去噪中：第 {done}/{total} 步 ({int(pct * 100)}%) · 預估剩餘約 {est_min_left} 分鐘",
            progress,
        )

    try:
        report_stage("[1/4] 正在使用 Qwen3-VL (8-bit) 進行提示詞文字編碼...", 0.05)
        media_result = pipeline.generate(
            config,
            paths,
            guard,
            on_step=on_step_progress,
            on_report=on_phase_report,
        )

        # Process subtitles if provided
        final_frames = media_result.frames
        clean_sub = subtitle_text.strip() if (subtitle_text and subtitle_text.strip()) else None
        if clean_sub:
            report_stage("Applying customized subtitle rendering...", 0.90)
            final_frames = render_subtitles_on_frames(
                media_result.frames,
                clean_sub,
                position=subtitle_position,
                style=subtitle_style,
                font_size=subtitle_font_size,
            )
            srt_filename = out_filename.rsplit(".", 1)[0] + ".srt"
            srt_path = target_out_dir / srt_filename
            write_srt_file(srt_path, clean_sub, actual_duration)

        report_stage("Phase 5/5: Muxing synchronized H.264 video and stereo audio...", 0.92)
        destination = output.mux_mp4(
            out_path,
            final_frames,
            media_result.audio,
            fps=media_result.fps,
            sample_rate=media_result.sample_rate,
        )

        elapsed_sec = time.perf_counter() - started_time
        peak_gib = round(peak_memory_bytes / (1024**3), 2) if peak_memory_bytes > 0 else round(mx.get_peak_memory() / (1024**3), 2)

        report_stage(f"Done! Output saved to: {destination} ({elapsed_sec/60:.1f} min, Peak RAM: {peak_gib} GB)", 1.0)

        res = GenerationResult(
            success=True,
            output_path=str(destination),
            output_filename=out_filename,
            output_dir=str(target_out_dir),
            prompt=prompt.strip(),
            seed=actual_seed,
            width=actual_width,
            height=actual_height,
            frames=actual_frames,
            duration_sec=actual_duration,
            steps=actual_steps,
            execution_time_sec=round(elapsed_sec, 1),
            peak_memory_gib=peak_gib,
            subtitle_text=clean_sub,
        )
        append_history(res)
        return res

    except Exception as e:
        elapsed_sec = time.perf_counter() - started_time
        error_msg = str(e)
        report_stage(f"Generation failed: {error_msg}", 1.0)

        # Write detailed failure log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Seed: {actual_seed}\n")
            f.write(f"Resolution: {actual_width}x{actual_height}\n")
            f.write(f"Frames: {actual_frames}\n")
            f.write(f"Steps: {actual_steps}\n")
            f.write(f"Elapsed: {elapsed_sec:.1f} s\n")
            f.write(f"Error: {error_msg}\n")

        res = GenerationResult(
            success=False,
            output_path="",
            output_filename="",
            output_dir=str(target_out_dir),
            prompt=prompt.strip(),
            seed=actual_seed,
            width=actual_width,
            height=actual_height,
            frames=actual_frames,
            duration_sec=actual_duration,
            steps=actual_steps,
            execution_time_sec=round(elapsed_sec, 1),
            peak_memory_gib=round(mx.get_peak_memory() / (1024**3), 2),
            error_message=error_msg,
        )
        append_history(res)
        return res
    finally:
        gc.collect()
        memory.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiniMax-H3 Engine CLI")
    parser.add_argument("prompt", nargs="?", help="Text prompt")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="fast")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    prof = PROFILES.get(args.profile, PROFILES["fast"])
    w = args.width or prof["width"]
    h = args.height or prof["height"]
    dur = args.duration or prof["duration_sec"]
    frames = args.frames or prof.get("frames")
    steps = args.steps or prof["steps"]

    p = args.prompt or "A cinematic shot of a happy golden retriever running in a park, sunlight filtering through trees, realistic motion"
    print(f"Starting generation with profile: {args.profile} ({w}x{h}, {dur}s, steps={steps})")
    result = generate_video(
        prompt=p,
        width=w,
        height=h,
        duration_sec=dur,
        frames=frames,
        seed=args.seed,
        steps=steps,
    )
    if result.success:
        print(f"\nSUCCESS: Video created at {result.output_path}")
        sys.exit(0)
    else:
        print(f"\nFAILED: {result.error_message}")
        sys.exit(1)
