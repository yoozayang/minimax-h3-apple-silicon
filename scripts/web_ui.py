#!/usr/bin/env python3
"""Unified Local Generative Studio (色色 Studio) for Apple Silicon.

Supports:
- Unified Prompt Workspace (Text -> Image, Text -> Video, Image -> Video, Reference -> Video)
- Native MLX Image Generation Engine (MFLUX / FLUX Schnell & Dev) with Dynamic Model Swap
- MiniMax-H3 33B DiT Video Generation (T2V, I2V Start Frame, Reference, Long Mode Continuation)
- Unified Asset Library & Batch Queue with Auto-continuation
- Post-processing Subtitle Editor (Pillow / FFmpeg)
- Dual View History Showcase (Grid / Compact List with Hide Item)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

# Dynamically resolve project directory
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_OUTPUT_DIR = OUTPUTS_DIR / "images"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"
QUEUE_FILE = LOGS_DIR / "queue.jsonl"
ASSETS_FILE = LOGS_DIR / "assets.jsonl"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import engine
import image_engine
import model_manager
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="色色 Studio - Local Generative Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Job State
CURRENT_JOB = {
    "is_running": False,
    "stage": "Idle",
    "progress": 0.0,
    "result": None,
    "error": None,
    "started_at": None,
    "prompt": "",
    "job_type": "VIDEO",  # "VIDEO" or "IMAGE"
    "width": 768,
    "height": 448,
    "duration_sec": 2.0,
    "steps": 10,
    "seed": -1,
    "output_dir": "",
    "active_queue_id": None,
    "cancel_event": None,
}
JOB_LOCK = threading.Lock()

# Prompt Queue State
AUTO_QUEUE_ENABLED = True
PROMPT_QUEUE: list[dict] = []
QUEUE_LOCK = threading.Lock()


def load_queue_from_file():
    global PROMPT_QUEUE
    if not QUEUE_FILE.exists():
        return
    with QUEUE_LOCK:
        PROMPT_QUEUE = []
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if item.get("status") == "running":
                                item["status"] = "queued"
                            PROMPT_QUEUE.append(item)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Warning: Failed to load queue: {e}", file=sys.stderr)


def save_queue_to_file():
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            for item in PROMPT_QUEUE:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to save queue: {e}", file=sys.stderr)


load_queue_from_file()


# Request Models
class ImageGenerateRequest(BaseModel):
    prompt: str
    width: int = 768
    height: int = 768
    steps: int = 4
    seed: int = -1
    model_name: str = "fhdr-uncensored"
    quality_profile: str = "high"
    extreme_quality_mode: bool = False
    quantize: int = 4
    count: int = 1
    output_dir: str | None = None


class HFTokenRequest(BaseModel):
    token: str


class GenerateRequest(BaseModel):
    prompt: str
    profile: str = "fast"
    mode: str = "text"  # "text", "image", "reference"
    start_image: str | None = None
    references: list | None = None
    long_mode: bool = False
    target_duration_sec: float | None = None
    width: int = 768
    height: int = 448
    duration_sec: float = 2.0
    seed: int = -1
    steps: int = 10
    output_dir: str = ""
    queue_id: str | None = None


class BatchAddRequest(BaseModel):
    prompts_text: str
    profile: str = "fast"
    mode: str = "text"
    start_image: str | None = None
    references: list | None = None
    width: int = 768
    height: int = 448
    duration_sec: float = 2.0
    steps: int = 10
    seed: int = -1
    output_dir: str = ""


class QueueActionRequest(BaseModel):
    item_id: str
    action: str  # retry, run_now, pause, resume, delete, move_top


class SubtitleBurnRequest(BaseModel):
    video_path: str
    text: str
    start_sec: float = 0.0
    end_sec: float | None = None
    position: str = "bottom"
    style: str = "box"
    font_size: int = 24


def summarize_error(err_str: str) -> str:
    """Format raw error into concise human-readable summary."""
    low = err_str.lower()
    if "victim of gpu error/recovery" in low or "insufficient memory" in low or "out of memory" in low:
        return "⚠️ 顯存爆量 (OOM) - 建議降低秒數 (2~3s) 或解析度"
    if "operation not permitted" in low:
        return "⚠️ 目錄權限受限 - 建議改用專案預設資料夾"
    if "task was cancelled" in low or "interrupted" in low:
        return "🛑 使用者手動取消"
    if "swap" in low:
        return "⚠️ 虛擬記憶體交換過載 - 建議減少併發或重啟"
    clean = err_str.replace("\n", " ").strip()
    return clean[:90] + ("..." if len(clean) > 90 else "")


def execute_generation_task(
    prompt: str,
    width: int,
    height: int,
    duration_sec: float,
    steps: int,
    seed: int,
    output_dir: str,
    queue_id: str | None = None,
    profile: str = "fast",
    mode: str = "text",
    start_image: str | None = None,
    references: list | None = None,
    long_mode: bool = False,
    target_duration_sec: float | None = None,
):
    global CURRENT_JOB
    cancel_event = threading.Event()
    with JOB_LOCK:
        CURRENT_JOB["is_running"] = True
        CURRENT_JOB["job_type"] = "VIDEO"
        CURRENT_JOB["stage"] = "Starting generation..."
        CURRENT_JOB["progress"] = 0.01
        CURRENT_JOB["result"] = None
        CURRENT_JOB["error"] = None
        CURRENT_JOB["started_at"] = time.time()
        CURRENT_JOB["prompt"] = prompt.strip()
        CURRENT_JOB["width"] = width
        CURRENT_JOB["height"] = height
        CURRENT_JOB["duration_sec"] = duration_sec
        CURRENT_JOB["steps"] = steps
        CURRENT_JOB["seed"] = seed
        CURRENT_JOB["output_dir"] = output_dir
        CURRENT_JOB["active_queue_id"] = queue_id
        CURRENT_JOB["cancel_event"] = cancel_event

    # Mark queue item as running if from queue
    if queue_id:
        with QUEUE_LOCK:
            for item in PROMPT_QUEUE:
                if item["id"] == queue_id:
                    item["status"] = "running"
                    item["error_message"] = None
                    break
            save_queue_to_file()

    def on_stage(stage_text: str, progress_val: float):
        with JOB_LOCK:
            CURRENT_JOB["stage"] = stage_text
            CURRENT_JOB["progress"] = progress_val

    try:
        w = width
        h = height
        dur = max(0.5, min(30.0, duration_sec))
        st = max(4, min(60, steps))
        seed_val = seed if seed >= 0 else None
        custom_out = output_dir.strip() if output_dir and output_dir.strip() else str(OUTPUTS_DIR)

        if long_mode and target_duration_sec and target_duration_sec > dur:
            res = engine.generate_long_video(
                prompt=prompt,
                target_duration_sec=target_duration_sec,
                segment_duration_sec=dur,
                width=w,
                height=h,
                seed=seed_val,
                steps=st,
                start_image=start_image if mode == "image" else None,
                references=references if mode == "reference" else None,
                output_dir=custom_out,
                cancel_event=cancel_event,
                on_stage=on_stage,
            )
        else:
            res = engine.generate_video(
                prompt=prompt,
                width=w,
                height=h,
                duration_sec=dur,
                seed=seed_val,
                steps=st,
                first_frame=start_image if mode == "image" else None,
                references=references if mode == "reference" else None,
                output_dir=custom_out,
                cancel_event=cancel_event,
                on_stage=on_stage,
            )

        if not res.success:
            if "cancelled" in (res.error_message or "").lower() or (cancel_event and cancel_event.is_set()):
                raise InterruptedError("Generation task was cancelled by user.")
            raise RuntimeError(res.error_message or "Generation failed")

        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["progress"] = 1.0
            CURRENT_JOB["stage"] = "Completed successfully!"
            CURRENT_JOB["result"] = engine.asdict(res)
            CURRENT_JOB["cancel_event"] = None

        if queue_id:
            with QUEUE_LOCK:
                for item in PROMPT_QUEUE:
                    if item["id"] == queue_id:
                        item["status"] = "completed"
                        item["output_path"] = res.output_path
                        break
                save_queue_to_file()

    except InterruptedError:
        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["stage"] = "Cancelled."
            CURRENT_JOB["error"] = "Generation was cancelled by user."
            CURRENT_JOB["progress"] = 0.0
            CURRENT_JOB["cancel_event"] = None

        with QUEUE_LOCK:
            target_item = None
            if queue_id:
                for idx, item in enumerate(PROMPT_QUEUE):
                    if item["id"] == queue_id:
                        target_item = PROMPT_QUEUE.pop(idx)
                        break
            if target_item is None:
                target_item = {
                    "id": str(uuid.uuid4())[:8],
                    "prompt": prompt,
                    "profile": profile,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                    "steps": steps,
                    "seed": seed,
                    "created_at": datetime.now().strftime("%H:%M:%S"),
                    "output_path": None,
                }
            target_item["status"] = "cancelled"
            target_item["error_message"] = "🛑 已手動取消"
            PROMPT_QUEUE.append(target_item)
            save_queue_to_file()

    except Exception as e:
        err_msg = str(e)
        friendly_err = summarize_error(err_msg)
        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["stage"] = "Generation failed."
            CURRENT_JOB["error"] = friendly_err
            CURRENT_JOB["progress"] = 0.0
            CURRENT_JOB["cancel_event"] = None

        with QUEUE_LOCK:
            target_item = None
            if queue_id:
                for idx, item in enumerate(PROMPT_QUEUE):
                    if item["id"] == queue_id:
                        target_item = PROMPT_QUEUE.pop(idx)
                        break
            if target_item is None:
                target_item = {
                    "id": str(uuid.uuid4())[:8],
                    "prompt": prompt,
                    "profile": profile,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                    "steps": steps,
                    "seed": seed,
                    "created_at": datetime.now().strftime("%H:%M:%S"),
                    "output_path": None,
                }
            target_item["status"] = "failed"
            target_item["error_message"] = friendly_err
            PROMPT_QUEUE.append(target_item)
            save_queue_to_file()


def queue_worker_loop():
    """Continuous background worker that triggers queued items when idle."""
    while True:
        time.sleep(1.5)
        if not AUTO_QUEUE_ENABLED:
            continue
        with JOB_LOCK:
            if CURRENT_JOB["is_running"]:
                continue

        next_item = None
        with QUEUE_LOCK:
            for item in PROMPT_QUEUE:
                if item.get("status") == "queued":
                    next_item = item
                    next_item["status"] = "running"
                    save_queue_to_file()
                    break
        if next_item:
            threading.Thread(
                target=execute_generation_task,
                kwargs={
                    "prompt": next_item["prompt"],
                    "width": next_item.get("width", 768),
                    "height": next_item.get("height", 448),
                    "duration_sec": next_item.get("duration_sec", 2.0),
                    "steps": next_item.get("steps", 10),
                    "seed": next_item.get("seed", -1),
                    "output_dir": next_item.get("output_dir", ""),
                    "queue_id": next_item["id"],
                    "profile": next_item.get("profile", "fast"),
                    "mode": next_item.get("mode", "text"),
                    "start_image": next_item.get("start_image"),
                    "references": next_item.get("references"),
                },
                daemon=True,
            ).start()


threading.Thread(target=queue_worker_loop, daemon=True).start()


# API Endpoints
@app.get("/api/status")
async def get_status():
    mem = engine.get_system_memory_status()
    with JOB_LOCK:
        job_info = {
            "is_running": CURRENT_JOB["is_running"],
            "job_type": CURRENT_JOB.get("job_type", "VIDEO"),
            "stage": CURRENT_JOB["stage"],
            "progress": round(CURRENT_JOB["progress"], 2),
            "elapsed_sec": round(time.time() - CURRENT_JOB["started_at"], 1) if CURRENT_JOB["started_at"] else 0,
            "has_result": CURRENT_JOB["result"] is not None,
            "error": CURRENT_JOB["error"],
            "prompt": CURRENT_JOB.get("prompt", ""),
            "active_queue_id": CURRENT_JOB.get("active_queue_id"),
        }
    return {
        "memory": mem,
        "job": job_info,
        "active_engine": model_manager.get_active_engine(),
        "profiles": engine.PROFILES,
        "default_output_dir": str(OUTPUTS_DIR),
        "desktop_dir": str(Path("~/Desktop").expanduser().resolve()),
        "downloads_dir": str(Path("~/Downloads").expanduser().resolve()),
        "movies_dir": str(Path("~/Movies").expanduser().resolve()),
        "auto_queue": AUTO_QUEUE_ENABLED,
    }


@app.get("/api/job")
async def get_job():
    with JOB_LOCK:
        elapsed = round(time.time() - CURRENT_JOB["started_at"], 1) if CURRENT_JOB["started_at"] else 0
        return {
            "is_running": CURRENT_JOB["is_running"],
            "job_type": CURRENT_JOB.get("job_type", "VIDEO"),
            "stage": CURRENT_JOB["stage"],
            "progress": round(CURRENT_JOB["progress"], 2),
            "elapsed_sec": elapsed,
            "prompt": CURRENT_JOB.get("prompt", ""),
            "result": CURRENT_JOB["result"],
            "error": CURRENT_JOB["error"],
            "active_queue_id": CURRENT_JOB.get("active_queue_id"),
        }


@app.get("/api/image/models")
async def get_image_models_endpoint():
    """Retrieve available image models, capabilities, and default selection."""
    return image_engine.get_available_image_models()


@app.post("/api/settings/hf-token")
async def set_hf_token_endpoint(req: HFTokenRequest):
    """Save user Hugging Face token to local environment and cache."""
    try:
        image_engine.save_hf_token(req.token)
        return {"status": "ok", "message": "Hugging Face Token 設定成功！"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/image/generate")
async def generate_image_endpoint(req: ImageGenerateRequest):
    """Generate image locally using MFLUX with dynamic memory swap."""
    with JOB_LOCK:
        if CURRENT_JOB["is_running"]:
            raise HTTPException(status_code=409, detail="已有生成任務正在執行中，請稍候。")
        CURRENT_JOB["is_running"] = True
        CURRENT_JOB["job_type"] = "IMAGE"
        CURRENT_JOB["stage"] = "正在載入圖片引擎並生成..."
        CURRENT_JOB["progress"] = 0.05
        CURRENT_JOB["started_at"] = time.time()
        CURRENT_JOB["prompt"] = req.prompt.strip()
        CURRENT_JOB["result"] = None
        CURRENT_JOB["error"] = None

    cancel_ev = threading.Event()
    with JOB_LOCK:
        CURRENT_JOB["cancel_event"] = cancel_ev

    def on_img_prog(prog: float, msg: str):
        with JOB_LOCK:
            CURRENT_JOB["progress"] = prog
            CURRENT_JOB["stage"] = msg

    try:
        results = image_engine.generate_images(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            seed=req.seed,
            model_name=req.model_name,
            quality_profile=req.quality_profile,
            extreme_quality_mode=req.extreme_quality_mode,
            quantize=req.quantize,
            count=req.count,
            output_dir=req.output_dir,
            progress_callback=on_img_prog,
            cancel_check=lambda: cancel_ev.is_set(),
        )
        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["progress"] = 1.0
            CURRENT_JOB["stage"] = "圖片生成完成！"
            CURRENT_JOB["result"] = [image_engine.asdict(r) for r in results]
            CURRENT_JOB["cancel_event"] = None
        return {"status": "ok", "results": [image_engine.asdict(r) for r in results]}

    except Exception as e:
        err_msg = str(e)
        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["stage"] = "圖片生成失敗"
            CURRENT_JOB["error"] = err_msg
            CURRENT_JOB["progress"] = 0.0
            CURRENT_JOB["cancel_event"] = None
        raise HTTPException(status_code=500, detail=f"圖片生成失敗: {err_msg}")


@app.get("/api/image/history")
async def get_image_history_endpoint():
    return image_engine.get_image_history(limit=40)


@app.post("/api/assets/upload")
async def upload_asset_endpoint(file: UploadFile = File(...)):
    """Accept real file upload from browser FormData, save to assets/uploads/, and return absolute path."""
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_filename = f"{time_str}_{file.filename}"
    upload_dir = ASSETS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / safe_filename
    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    rec = image_engine.register_asset(
        asset_type="IMAGE",
        source="UPLOAD",
        file_path=target_path,
        prompt="",
    )
    return {"status": "ok", "asset": rec, "path": str(target_path)}


@app.get("/api/assets")
async def get_assets_endpoint():
    if not ASSETS_FILE.exists():
        return []
    records = []
    with open(ASSETS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except Exception:
                    pass
    return records[::-1]


@app.post("/api/generate")
async def start_generation(req: GenerateRequest):
    with JOB_LOCK:
        if CURRENT_JOB["is_running"]:
            raise HTTPException(status_code=409, detail="已有生成任務正在執行中，請稍候或加入排程。")
    thread = threading.Thread(
        target=execute_generation_task,
        kwargs={
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "duration_sec": req.duration_sec,
            "steps": req.steps,
            "seed": req.seed,
            "output_dir": req.output_dir,
            "queue_id": req.queue_id,
            "profile": req.profile,
            "mode": req.mode,
            "start_image": req.start_image,
            "references": req.references,
            "long_mode": req.long_mode,
            "target_duration_sec": req.target_duration_sec,
        },
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@app.post("/api/generate/cancel")
async def cancel_generation():
    with JOB_LOCK:
        if not CURRENT_JOB["is_running"] or not CURRENT_JOB.get("cancel_event"):
            return {"status": "not_running"}
        CURRENT_JOB["cancel_event"].set()
        CURRENT_JOB["stage"] = "正在中斷任務並釋放顯存..."
    return {"status": "cancelling"}


@app.get("/api/queue")
async def get_queue():
    with QUEUE_LOCK:
        return {"items": PROMPT_QUEUE, "auto_enabled": AUTO_QUEUE_ENABLED}


@app.post("/api/queue/batch-add")
async def batch_add_queue(req: BatchAddRequest):
    lines = [l.strip() for l in req.prompts_text.split("\n") if l.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="請輸入至少一筆提示詞")
    added_items = []
    with QUEUE_LOCK:
        for prompt_line in lines:
            item_id = str(uuid.uuid4())[:8]
            queue_item = {
                "id": item_id,
                "prompt": prompt_line,
                "profile": req.profile,
                "mode": req.mode,
                "start_image": req.start_image,
                "references": req.references,
                "width": req.width,
                "height": req.height,
                "duration_sec": req.duration_sec,
                "steps": req.steps,
                "seed": req.seed,
                "output_dir": req.output_dir,
                "status": "queued",
                "error_message": None,
                "created_at": datetime.now().strftime("%H:%M:%S"),
                "output_path": None,
            }
            PROMPT_QUEUE.append(queue_item)
            added_items.append(queue_item)
        save_queue_to_file()
    return {"status": "ok", "added_count": len(added_items), "items": PROMPT_QUEUE}


@app.post("/api/queue/action")
async def handle_queue_action(req: QueueActionRequest):
    with QUEUE_LOCK:
        target_idx = None
        for idx, item in enumerate(PROMPT_QUEUE):
            if item["id"] == req.item_id:
                target_idx = idx
                break
        if target_idx is None:
            raise HTTPException(status_code=404, detail="找不到指定的排程項目")

        item = PROMPT_QUEUE[target_idx]
        if req.action == "delete":
            PROMPT_QUEUE.pop(target_idx)
        elif req.action == "retry":
            item["status"] = "queued"
            item["error_message"] = None
        elif req.action == "pause":
            item["status"] = "paused"
        elif req.action == "resume":
            item["status"] = "queued"
        elif req.action == "move_top":
            popped = PROMPT_QUEUE.pop(target_idx)
            popped["status"] = "queued"
            PROMPT_QUEUE.insert(0, popped)
        elif req.action == "run_now":
            with JOB_LOCK:
                if CURRENT_JOB["is_running"]:
                    raise HTTPException(status_code=409, detail="目前已有任務正在運行，請等待或先中止。")
            popped = PROMPT_QUEUE.pop(target_idx)
            popped["status"] = "running"
            PROMPT_QUEUE.insert(0, popped)
            save_queue_to_file()
            threading.Thread(
                target=execute_generation_task,
                kwargs={
                    "prompt": popped["prompt"],
                    "width": popped.get("width", 768),
                    "height": popped.get("height", 448),
                    "duration_sec": popped.get("duration_sec", 2.0),
                    "steps": popped.get("steps", 10),
                    "seed": popped.get("seed", -1),
                    "output_dir": popped.get("output_dir", ""),
                    "queue_id": popped["id"],
                    "profile": popped.get("profile", "fast"),
                    "mode": popped.get("mode", "text"),
                    "start_image": popped.get("start_image"),
                    "references": popped.get("references"),
                },
                daemon=True,
            ).start()
            return {"status": "started", "items": PROMPT_QUEUE}

        save_queue_to_file()
    return {"status": "ok", "items": PROMPT_QUEUE}


@app.post("/api/queue/toggle-auto")
async def toggle_auto_queue():
    global AUTO_QUEUE_ENABLED
    AUTO_QUEUE_ENABLED = not AUTO_QUEUE_ENABLED
    return {"status": "ok", "auto_enabled": AUTO_QUEUE_ENABLED}


@app.post("/api/subtitles/burn")
async def burn_subtitles_endpoint(req: SubtitleBurnRequest):
    """Post-processing subtitle burn-in onto existing video file."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="請輸入字幕文字")
    v_path = Path(req.video_path).expanduser().resolve()
    if not v_path.exists():
        raise HTTPException(status_code=404, detail="找不到來源影片檔案")

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = f"{v_path.stem}_sub_{time_str}.mp4"
    out_path = v_path.parent / out_name

    try:
        dest = engine.burn_subtitles_to_video_file(
            input_video_path=v_path,
            output_video_path=out_path,
            subtitle_text=req.text.strip(),
            start_sec=req.start_sec,
            end_sec=req.end_sec,
            position=req.position,
            style=req.style,
            font_size=req.font_size,
        )
        return {
            "status": "ok",
            "output_path": str(dest),
            "output_filename": out_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"字幕壓制失敗: {e}")


@app.get("/api/history")
async def get_history():
    return engine.get_history(limit=40)


@app.get("/api/video-stream")
async def stream_video(path: str = Query(...)):
    """Serve media directly from any local folder with standard headers."""
    decoded_path = urllib.parse.unquote(path)
    file_path = Path(decoded_path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="檔案不存在")
    import mimetypes
    mime, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        path=str(file_path),
        media_type=mime or "application/octet-stream",
        filename=file_path.name,
    )


@app.post("/api/open-folder")
async def open_output_folder(data: dict):
    file_path = data.get("file_path")
    dir_path = data.get("dir_path")
    try:
        if file_path and Path(file_path).exists():
            subprocess.run(["open", "-R", str(file_path)], check=True)
        elif dir_path and Path(dir_path).exists():
            subprocess.run(["open", str(dir_path)], check=True)
        else:
            subprocess.run(["open", str(OUTPUTS_DIR)], check=True)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


# Complete SPA HTML
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-TW" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>色色 Studio - 本機生成創作工作站</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f17;
      --card-bg: rgba(20, 27, 41, 0.8);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary-accent: #6366f1;
      --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
      --image-gradient: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
      --danger-gradient: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(236, 72, 153, 0.10) 0%, transparent 40%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      background: rgba(11, 15, 23, 0.85);
      border-bottom: 1px solid var(--card-border);
      padding: 0.85rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand { display: flex; align-items: center; gap: 0.75rem; }
    .brand-badge {
      background: var(--primary-gradient);
      width: 38px; height: 38px; border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 800; font-size: 1.2rem; color: white;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    .brand-text h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.5px; }
    .brand-text p { font-size: 0.75rem; color: var(--text-muted); }
    .system-status { display: flex; align-items: center; gap: 0.75rem; font-size: 0.8rem; }
    .status-pill {
      background: var(--card-bg); border: 1px solid var(--card-border);
      padding: 0.35rem 0.85rem; border-radius: 9999px;
      display: flex; align-items: center; gap: 0.5rem;
      font-family: 'JetBrains Mono', monospace;
    }
    .indicator-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--success); box-shadow: 0 0 8px var(--success);
    }
    .indicator-dot.warning { background: var(--warning); box-shadow: 0 0 8px var(--warning); }
    .indicator-dot.danger { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

    main {
      flex: 1; max-width: 1400px; margin: 0 auto;
      padding: 1.5rem; width: 100%;
      display: grid; grid-template-columns: 1.1fr 1fr; gap: 1.5rem;
    }
    @media (max-width: 1080px) { main { grid-template-columns: 1fr; } }

    .glass-card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.25rem;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      display: flex; flex-direction: column; gap: 1rem;
    }

    .card-title {
      font-size: 1.05rem; font-weight: 700;
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--card-border); padding-bottom: 0.6rem;
    }

    /* Mode Navigation Tabs */
    .mode-nav {
      display: flex; gap: 0.5rem; background: rgba(11, 15, 23, 0.6);
      padding: 0.3rem; border-radius: 10px; border: 1px solid var(--card-border);
    }
    .mode-tab {
      flex: 1; text-align: center; padding: 0.45rem 0.6rem; border-radius: 8px;
      font-size: 0.8rem; font-weight: 600; cursor: pointer; color: var(--text-muted);
      transition: all 0.2s ease; border: 1px solid transparent;
    }
    .mode-tab:hover { color: var(--text-main); }
    .mode-tab.active {
      background: rgba(99, 102, 241, 0.25); color: #c7d2fe;
      border-color: rgba(99, 102, 241, 0.4); box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
    label { font-size: 0.8rem; font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; }
    textarea, select, input {
      background: rgba(11, 15, 23, 0.7); border: 1px solid var(--card-border);
      color: var(--text-main); padding: 0.65rem 0.85rem; border-radius: 10px;
      font-family: inherit; font-size: 0.9rem; outline: none; transition: 0.2s ease;
    }
    textarea:focus, select:focus, input:focus {
      border-color: var(--primary-accent); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
    }
    textarea { min-height: 80px; resize: vertical; line-height: 1.45; }

    /* Action Buttons Bar */
    .btn-action-row {
      display: grid; grid-template-columns: 1fr 1.3fr 1fr; gap: 0.6rem;
    }
    .btn-gen-img {
      background: var(--image-gradient); border: none; color: white;
      font-weight: 700; padding: 0.75rem; border-radius: 10px;
      cursor: pointer; transition: 0.2s ease; font-size: 0.85rem;
      box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3);
      display: flex; justify-content: center; align-items: center; gap: 0.35rem;
    }
    .btn-gen-vid {
      background: var(--primary-gradient); border: none; color: white;
      font-weight: 700; padding: 0.75rem; border-radius: 10px;
      cursor: pointer; transition: 0.2s ease; font-size: 0.9rem;
      box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35);
      display: flex; justify-content: center; align-items: center; gap: 0.4rem;
    }
    .btn-add-q {
      background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.15);
      color: var(--text-main); font-weight: 600; padding: 0.75rem; border-radius: 10px;
      cursor: pointer; transition: 0.2s ease; font-size: 0.85rem;
      display: flex; justify-content: center; align-items: center; gap: 0.35rem;
    }
    .btn-gen-img:hover, .btn-gen-vid:hover, .btn-add-q:hover {
      transform: translateY(-1px); filter: brightness(1.08);
    }

    /* Start Frame & Reference Dropzones */
    .dropzone-box {
      border: 2px dashed var(--card-border); border-radius: 10px;
      padding: 0.85rem; background: rgba(11, 15, 23, 0.4); text-align: center;
      cursor: pointer; transition: 0.2s ease; display: flex; align-items: center; gap: 0.75rem;
    }
    .dropzone-box:hover { border-color: var(--primary-accent); background: rgba(99, 102, 241, 0.05); }
    .dropzone-preview {
      width: 64px; height: 64px; border-radius: 8px; object-fit: cover;
      background: #1f2937; border: 1px solid var(--card-border);
    }

    /* Generated Images Gallery */
    .images-gallery-wrap {
      display: flex; flex-direction: column; gap: 0.5rem;
      background: rgba(11, 15, 23, 0.5); border: 1px solid var(--card-border);
      border-radius: 10px; padding: 0.75rem;
    }
    .images-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 0.6rem;
    }
    .img-card {
      background: rgba(255, 255, 255, 0.03); border: 1px solid var(--card-border);
      border-radius: 8px; overflow: hidden; display: flex; flex-direction: column;
      transition: 0.2s ease;
    }
    .img-card:hover { border-color: #818cf8; transform: translateY(-2px); }
    .img-card img {
      width: 100%; aspect-ratio: 1/1; object-fit: cover; cursor: pointer;
    }
    .img-card-actions {
      display: flex; flex-direction: column; gap: 0.2rem; padding: 0.35rem; background: rgba(11, 15, 23, 0.85);
    }
    .btn-card-mini {
      background: rgba(255, 255, 255, 0.06); border: 1px solid var(--card-border);
      color: var(--text-muted); font-size: 0.68rem; padding: 0.25rem 0.3rem; border-radius: 4px;
      cursor: pointer; text-align: center; transition: 0.15s ease;
    }
    .btn-card-mini:hover { background: rgba(99, 102, 241, 0.25); color: #fff; border-color: var(--primary-accent); }

    /* Progress & Cancel Mini Button */
    .progress-card {
      background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1rem;
      display: flex; flex-direction: column; gap: 0.6rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .progress-bar-bg {
      background: rgba(255, 255, 255, 0.08); height: 8px; border-radius: 9999px; overflow: hidden;
    }
    .progress-fill {
      background: var(--primary-gradient); height: 100%; width: 0%;
      border-radius: 9999px; transition: width 0.3s ease;
    }
    .btn-cancel-mini {
      background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5; font-size: 0.75rem; font-weight: 700; padding: 0.35rem 0.75rem;
      border-radius: 6px; cursor: pointer; transition: 0.2s ease;
    }
    .btn-cancel-mini:hover { background: var(--danger-gradient); color: white; }

    /* Queue & History */
    .queue-card {
      background: rgba(11, 15, 23, 0.6); border: 1px solid var(--card-border);
      border-radius: 8px; padding: 0.65rem 0.85rem; display: flex; flex-direction: column;
      gap: 0.4rem; transition: 0.2s ease;
    }
    .queue-card.running { border-color: var(--primary-accent); background: rgba(99, 102, 241, 0.08); }
    .queue-card.completed { border-color: rgba(16, 185, 129, 0.3); }
    .queue-header { display: flex; justify-content: space-between; align-items: center; }
    .badge-status {
      font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600;
    }
    .badge-status.queued { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .badge-status.running { background: rgba(99, 102, 241, 0.2); color: #818cf8; }
    .badge-status.completed { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .badge-status.failed { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .badge-status.cancelled { background: rgba(156, 163, 175, 0.15); color: #9ca3af; }

    .btn-tiny {
      background: rgba(255, 255, 255, 0.05); border: 1px solid var(--card-border);
      color: var(--text-muted); font-size: 0.7rem; padding: 0.2rem 0.45rem; border-radius: 4px;
      cursor: pointer; transition: 0.15s ease;
    }
    .btn-tiny:hover { background: rgba(255, 255, 255, 0.15); color: var(--text-main); }

    /* Video Player */
    .player-container {
      background: #000; border-radius: 12px; overflow: hidden;
      aspect-ratio: 16/9; position: relative; border: 1px solid var(--card-border);
      display: flex; align-items: center; justify-content: center;
    }
    video { width: 100%; height: 100%; object-fit: contain; }
    .empty-state { text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 2rem; }

    /* Collapsible */
    .collapsible {
      border: 1px solid var(--card-border); border-radius: 10px;
      overflow: hidden; background: rgba(11, 15, 23, 0.4);
    }
    .collapsible-header {
      padding: 0.65rem 0.85rem; cursor: pointer; display: flex;
      justify-content: space-between; align-items: center;
      font-size: 0.8rem; font-weight: 600; color: var(--text-muted);
    }
    .collapsible-content { padding: 0.85rem; display: none; flex-direction: column; gap: 0.75rem; border-top: 1px solid var(--card-border); }
    .collapsible.open .collapsible-content { display: flex; }

    /* History Section Styles */
    .history-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 1rem;
      width: 100%;
    }
    .history-item {
      position: relative;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--card-border);
      background: rgba(11, 15, 23, 0.7);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      transition: 0.2s ease;
    }
    .history-item:hover {
      border-color: var(--primary-accent);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(99,102,241,0.25);
    }
    .history-video-thumb {
      aspect-ratio: 16/9;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
    }
    .history-video-thumb video {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .history-prompt {
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--text-main);
      padding: 0.6rem 0.75rem 0.3rem 0.75rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .history-footer {
      font-size: 0.7rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      padding: 0 0.75rem 0.6rem 0.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .btn-hide-item {
      position: absolute;
      top: 6px;
      right: 6px;
      z-index: 10;
      background: rgba(0, 0, 0, 0.75);
      color: #fff;
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 50%;
      width: 22px;
      height: 22px;
      font-size: 11px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      transition: 0.2s ease;
    }
    .history-item:hover .btn-hide-item {
      opacity: 1;
    }
    .history-list-view {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      width: 100%;
    }
    .history-list-row {
      background: rgba(11, 15, 23, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 0.6rem 0.85rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      transition: 0.15s ease;
    }
    .history-list-row:hover {
      border-color: var(--primary-accent);
      background: rgba(99, 102, 241, 0.08);
    }
    .chips-container { display: flex; align-items: center; gap: 0.4rem; }
    .chip {
      font-size: 0.75rem; padding: 0.2rem 0.55rem; border-radius: 6px;
      background: rgba(255,255,255,0.05); border: 1px solid var(--card-border);
      color: var(--text-muted); cursor: pointer; transition: 0.15s;
    }
    .chip.active { background: rgba(99,102,241,0.25); color: #a5b4fc; border-color: rgba(99,102,241,0.4); font-weight:600; }

    .toast {
      position: fixed; bottom: 2rem; right: 2rem;
      background: rgba(15, 23, 42, 0.95); border: 1px solid var(--card-border);
      color: white; padding: 0.75rem 1.25rem; border-radius: 10px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5); z-index: 100;
      display: none; font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-badge">色</div>
      <div class="brand-text">
        <h1>色色 Studio</h1>
        <p>Local Generative Studio · Apple Silicon MLX</p>
      </div>
    </div>
    <div class="system-status">
      <div class="status-pill">
        <span>RAM:</span>
        <span id="mem-used" style="color:#818cf8;">--</span> / <span id="mem-total">-- GB</span>
      </div>
      <div class="status-pill">
        <div id="status-indicator" class="indicator-dot"></div>
        <span id="system-state-text">Ready</span>
      </div>
    </div>
  </header>

  <main style="display:flex; flex-direction:column; gap:1.5rem;">
    <!-- Top 2-Column Main Workspace -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1.25rem;">
      <!-- Left Column: Unified Workspace -->
      <div style="display:flex; flex-direction:column; gap:1.25rem;">
        <!-- Unified Prompt Box -->
        <div class="glass-card">
          <div class="card-title">
            <span>✨ 統一提示詞工作區 (Unified Workspace)</span>
            <span style="font-size:0.75rem; font-weight:400; color:var(--text-muted);">支援生圖 · 生影片 · 圖生影</span>
          </div>

          <!-- Video Conditioning Mode Switcher -->
          <div class="mode-nav">
            <div class="mode-tab active" id="tab-mode-text" onclick="switchVideoMode('text')">📝 純文字生影片</div>
            <div class="mode-tab" id="tab-mode-image" onclick="switchVideoMode('image')">🖼️ 首幀圖生影片 (I2V)</div>
            <div class="mode-tab" id="tab-mode-reference" onclick="switchVideoMode('reference')">🏷️ 參考特徵 (Ref)</div>
          </div>

          <!-- Start Image Panel (for I2V mode) -->
          <div id="start-image-panel" style="display:none; flex-direction:column; gap:0.5rem;">
            <label>🎬 影片起始幀 (Start Frame Image) <span style="font-size:0.7rem; color:#818cf8;">首幀像素注入 H3 FL2VA</span></label>
            <div class="dropzone-box" onclick="document.getElementById('start-image-file').click();">
              <img id="start-image-thumb" class="dropzone-preview" src="" style="display:none;" />
              <div id="start-image-placeholder" style="flex:1;">
                <span style="font-size:0.8rem; color:var(--text-muted);">點擊或拖曳上傳圖片 (或從下方已生成圖片點選「設為起始幀」)</span>
              </div>
              <button class="btn-tiny" id="btn-clear-start-image" style="display:none;" onclick="event.stopPropagation(); clearStartImage();">✕ 清除</button>
              <input type="file" id="start-image-file" accept="image/*" style="display:none;" onchange="handleStartImageUpload(event);" />
            </div>
          </div>

          <!-- Reference Subjects Panel (for Reference mode) -->
          <div id="reference-panel" style="display:none; flex-direction:column; gap:0.5rem;">
            <label>🏷️ 角色 / 物件參考特徵 (Reference Subjects) <span style="font-size:0.7rem; color:#a855f7;">H3 多模態條件注入</span></label>
            <div style="background:rgba(11,15,23,0.5); padding:0.6rem; border-radius:8px; border:1px solid var(--card-border); display:flex; flex-direction:column; gap:0.4rem;">
              <div style="display:flex; gap:0.5rem; align-items:center;">
                <input type="text" id="ref-subject-name" placeholder="主體名稱 (例如: Money / 女主角 / 跑車)" style="flex:1; font-size:0.8rem;" />
                <button class="btn-tiny" onclick="addRefSubjectImage();">➕ 加圖</button>
              </div>
              <div id="ref-images-list" style="display:flex; gap:0.4rem; flex-wrap:wrap;"></div>
            </div>
          </div>

          <!-- Prompt Textarea -->
          <div class="form-group">
            <label for="prompt">
              <span>創意提示詞 (Prompt)</span>
              <span style="font-size:0.7rem; color:var(--text-muted);">多行自動支援批次排程</span>
            </label>
            <textarea id="prompt" placeholder="輸入描述 (例如: A cinematic shot of a happy golden retriever running in a park, soft sunlight filtering through trees)"></textarea>
          </div>

          <!-- Image Generation Controls (Model & Quality Bar) -->
          <div class="image-model-ctrl" style="display:flex; justify-content:space-between; align-items:center; background:rgba(11,15,23,0.6); padding:0.5rem 0.75rem; border-radius:8px; border:1px solid var(--card-border); margin-bottom:0.6rem; gap:0.6rem; flex-wrap:wrap;">
            <div style="display:flex; align-items:center; gap:0.4rem;">
              <span style="font-size:0.75rem; font-weight:700; color:#818cf8;">🎨 模型:</span>
              <select id="image-model-select" style="font-size:0.75rem; padding:0.25rem 0.5rem; background:rgba(0,0,0,0.5); border:1px solid var(--card-border); color:var(--text-main); border-radius:6px; font-weight:600;">
                <option value="fhdr-uncensored" selected>🌟 FHDR Uncensored — Quality</option>
                <option value="flux2-klein-4b">⚡ FLUX.2 Klein 4B — Fast</option>
              </select>
            </div>
            <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:nowrap;">
              <span style="font-size:0.75rem; font-weight:700; color:#a5b4fc;">品質:</span>
              <div class="chips-container" id="img-quality-selector" style="display:flex; gap:0.25rem;">
                <span class="chip" id="qchip-draft" onclick="setImageQuality('draft')">Draft</span>
                <span class="chip" id="qchip-balanced" onclick="setImageQuality('balanced')">Balanced</span>
                <span class="chip active" id="qchip-high" onclick="setImageQuality('high')">High</span>
                <span class="chip" id="qchip-maximum" onclick="setImageQuality('maximum')">Maximum</span>
              </div>
              <button id="btn-extreme-quality" class="btn-tiny" style="background:rgba(239,68,68,0.08); border-color:rgba(239,68,68,0.3); color:#fca5a5; font-weight:700; padding:0.25rem 0.55rem; border-radius:6px; cursor:pointer; transition:0.2s; white-space:nowrap; margin-left:0.2rem;" onclick="toggleExtremeQuality()" title="🔥 極限品質模式：不計耗時追求最高畫質（預設關閉）">🔥 極限品質: OFF</button>
            </div>
          </div>

          <!-- Action Buttons Bar -->
          <div class="btn-action-row">
            <button class="btn-gen-img" onclick="handleGenerateImageClick();" title="使用所選模型快速生成圖片">
              <span>🖼️ 生成圖片</span>
            </button>
            <button class="btn-gen-vid" id="btn-gen-video" onclick="handleGenerateVideoClick();" title="立即開始生成影片">
              <span>🚀 生成影片</span>
            </button>
            <button class="btn-add-q" onclick="handleAddToQueueClick();" title="將提示詞加入佇列排程">
              <span>➕ 加入排程</span>
            </button>
          </div>

          <!-- Active Progress Card (Prominently placed below action buttons) -->
          <div class="progress-card" id="progress-card" style="display:none; margin-top:0.4rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div style="display:flex; align-items:center; gap:0.5rem;">
                <span class="indicator-dot"></span>
                <span style="font-size:0.85rem; font-weight:700; color:#a5b4fc;" id="progress-stage-text">正在初始化...</span>
              </div>
              <div style="display:flex; align-items:center; gap:0.6rem;">
                <span style="font-size:0.8rem; font-family:'JetBrains Mono',monospace; color:var(--text-muted);" id="progress-time-text">⏱️ 0s</span>
                <span style="font-size:0.95rem; font-family:'JetBrains Mono',monospace; font-weight:700; color:#818cf8;" id="progress-pct-text">0%</span>
                <button class="btn-cancel-mini" onclick="cancelCurrentTask();" title="立即安全中斷並釋放顯存">🛑 中止生成</button>
              </div>
            </div>
            <div class="progress-bar-bg" style="height:10px;">
              <div class="progress-fill" id="progress-fill"></div>
            </div>
          </div>

          <!-- Generated Images Gallery -->
          <div class="images-gallery-wrap" id="images-gallery-wrap">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span style="font-size:0.8rem; font-weight:700;">🖼️ 本機生成圖片庫 (Image Gallery)</span>
              <button class="btn-tiny" onclick="fetchImageHistory();">🔄 重新整理</button>
            </div>
            <div class="images-grid" id="images-grid">
              <span style="font-size:0.75rem; color:var(--text-muted);">尚無生成圖片，點擊「生成圖片」立即探索構圖！</span>
            </div>
          </div>

          <!-- Video Profile & Settings Drawer -->
          <div class="collapsible open">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
              <span>⚙️ 影片生成規格與長度設定</span>
              <span>▼</span>
            </div>
            <div class="collapsible-content">
              <div class="form-group">
                <label>影片規格檔位 (Profile)</label>
                <select id="video-profile" onchange="handleProfileChange(this.value)">
                  <option value="fast" selected>⚡ 快速預覽 (768x448 · ~2.0s · 10 Steps)</option>
                  <option value="standard">🎬 標準長度 (960x544 · ~2.5s · 15 Steps)</option>
                  <option value="hd720">🌟 720p 高畫質 (1280x720 · ~2.0s · 15 Steps)</option>
                  <option value="long">📽️ 長影片分段延續模式 (Long Continuation)</option>
                  <option value="custom">🛠️ 自訂參數 (Custom)</option>
                </select>
              </div>

              <!-- Long Mode Target Duration -->
              <div id="long-mode-box" style="display:none; flex-direction:column; gap:0.4rem; background:rgba(99,102,241,0.08); padding:0.6rem; border-radius:8px; border:1px solid rgba(99,102,241,0.25);">
                <label style="color:#a5b4fc;">📽️ 長影片目標長度 (Target Duration)</label>
                <select id="long-target-duration" style="font-family:'JetBrains Mono',monospace;">
                  <option value="5.0">5.0 秒 (約 2 個分段)</option>
                  <option value="10.0" selected>10.0 秒 (約 4~5 個分段)</option>
                  <option value="15.0">15.0 秒 (約 6~7 個分段)</option>
                  <option value="30.0">30.0 秒 (極長敘事鏡頭)</option>
                </select>
              </div>

              <div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.6rem;">
                <div class="form-group">
                  <label>解析度寬高</label>
                  <div style="display:flex; gap:0.3rem;">
                    <input type="number" id="custom-w" value="768" style="width:50%;" />
                    <input type="number" id="custom-h" value="448" style="width:50%;" />
                  </div>
                </div>
                <div class="form-group">
                  <label>採樣步數 (Steps)</label>
                  <input type="number" id="custom-steps" value="10" min="4" max="60" />
                </div>
              </div>

              <!-- Output Directory Selector -->
              <div class="form-group" style="margin-top:0.4rem; padding-top:0.6rem; border-top:1px solid rgba(255,255,255,0.06);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                  <label style="margin-bottom:0; color:var(--text-main); font-weight:600;">📁 產出儲存位置 (Output Folder)</label>
                  <button class="btn-tiny" onclick="openCurrentFolder()" title="在 Finder 中開啟此資料夾">📂 在 Finder 開啟</button>
                </div>
                <div style="display:flex; gap:0.4rem; align-items:center;">
                  <input type="text" id="output-dir" class="path-input" placeholder="/Users/.../outputs" onchange="saveOutputDir(this.value)" style="flex:1; font-family:'JetBrains Mono',monospace; font-size:0.75rem; padding:0.45rem 0.65rem; border-radius:6px; background:rgba(0,0,0,0.3); border:1px solid var(--card-border); color:var(--text-main);" />
                </div>
                <div class="chips-container" id="path-presets" style="margin-top:0.4rem; flex-wrap:wrap;">
                  <span class="chip active" id="chip-default" onclick="selectPathPreset('default')">📁 專案預設 (outputs)</span>
                  <span class="chip" id="chip-desktop" onclick="selectPathPreset('desktop')">🖥️ 桌面 (Desktop)</span>
                  <span class="chip" id="chip-downloads" onclick="selectPathPreset('downloads')">📥 下載 (Downloads)</span>
                  <span class="chip" id="chip-movies" onclick="selectPathPreset('movies')">🎬 影片 (Movies)</span>
                </div>
              </div>

              <!-- Hugging Face Access Token Box -->
              <div class="form-group" style="margin-top:0.4rem; padding-top:0.6rem; border-top:1px solid rgba(255,255,255,0.06);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                  <label style="margin-bottom:0; color:var(--text-main); font-weight:600;">🔑 Hugging Face Token (Gated 授權模型專用)</label>
                </div>
                <div style="display:flex; gap:0.4rem; align-items:center;">
                  <input type="password" id="hf-token-input" placeholder="hf_..." style="flex:1; font-family:'JetBrains Mono',monospace; font-size:0.75rem; padding:0.45rem 0.65rem; border-radius:6px; background:rgba(0,0,0,0.3); border:1px solid var(--card-border); color:var(--text-main);" />
                  <button class="btn-tiny" onclick="saveHFTokenClick()" style="font-weight:700;">💾 儲存</button>
                </div>
                <span style="font-size:0.68rem; color:var(--text-muted); margin-top:0.2rem;">使用 FHDR-Uncensored-MFLUX 等 Gated 模型時，請填入具備 Read 權限之 HF Token。</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Prompt Queue Drawer -->
        <div class="glass-card">
          <div class="card-title">
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <span>📋 排程隊列 (Queue)</span>
              <span id="queue-count-badge" style="font-size:0.7rem; background:rgba(99,102,241,0.2); color:#a5b4fc; padding:0.15rem 0.5rem; border-radius:9999px;">0 筆</span>
            </div>
            <div style="display:flex; gap:0.4rem; align-items:center;">
              <label style="font-size:0.75rem; cursor:pointer; display:flex; align-items:center; gap:0.3rem;">
                <input type="checkbox" id="auto-queue-toggle" checked onchange="toggleAutoQueue();" />
                <span>🔁 自動連續生成</span>
              </label>
            </div>
          </div>
          <div id="queue-list" style="display:flex; flex-direction:column; gap:0.5rem; max-height:280px; overflow-y:auto;">
            <span style="font-size:0.75rem; color:var(--text-muted);">目前無排程工作。</span>
          </div>
        </div>
      </div>

      <!-- Right Column: Player & Subtitles -->
      <div style="display:flex; flex-direction:column; gap:1.25rem;">
        <!-- Player Wrap -->
        <div class="glass-card" id="player-wrap">
          <div class="card-title">
            <span>🎬 預覽與播放中心</span>
            <div style="display:flex; gap:0.4rem;" id="player-toolbar">
              <select id="play-speed" style="font-size:0.75rem; padding:0.2rem 0.4rem;" onchange="setPlaySpeed(this.value)">
                <option value="0.5">0.5x</option>
                <option value="1.0" selected>1.0x</option>
                <option value="1.5">1.5x</option>
                <option value="2.0">2.0x</option>
              </select>
              <button class="btn-tiny" id="btn-loop" onclick="toggleLoop()">🔁 循環</button>
            </div>
          </div>

          <div class="player-container">
            <div class="empty-state" id="empty-state">
              <div style="font-size:2.2rem; margin-bottom:0.5rem;">📽️</div>
              <div>目前尚無載入影片</div>
              <div style="font-size:0.75rem; margin-top:0.3rem;">點擊排程卡片、歷史紀錄或開始生成</div>
            </div>
            <video id="video-player" controls autoplay loop playsinline style="display:none;"></video>
          </div>

          <!-- Video Metadata Grid -->
          <div id="meta-grid" style="display:none; grid-template-columns: repeat(4, 1fr); gap:0.4rem; font-size:0.75rem; font-family:'JetBrains Mono',monospace; background:rgba(11,15,23,0.6); padding:0.6rem; border-radius:8px;">
            <div><span style="color:var(--text-muted);">解析度:</span> <span id="meta-res">--</span></div>
            <div><span style="color:var(--text-muted);">時長:</span> <span id="meta-dur">--</span></div>
            <div><span style="color:var(--text-muted);">Seed:</span> <span id="meta-seed">--</span></div>
            <div><span style="color:var(--text-muted);">耗時:</span> <span id="meta-time">--</span></div>
          </div>

          <!-- Subtitle Editor Drawer -->
          <div class="collapsible open" id="subtitles-drawer">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
              <span>💬 後期字幕編輯 (秒級快速壓制另存)</span>
              <span>▼</span>
            </div>
            <div class="collapsible-content">
              <div class="form-group">
                <input type="text" id="sub-text" placeholder="輸入要插入的字幕內容..." />
              </div>
              <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:0.5rem;">
                <div class="form-group">
                  <label>樣式</label>
                  <select id="sub-style">
                    <option value="box" selected>🔳 半透明黑底框</option>
                    <option value="classic">🔲 經典白字黑邊</option>
                    <option value="highlight">🟨 高對比亮黃</option>
                  </select>
                </div>
                <div class="form-group">
                  <label>位置</label>
                  <select id="sub-pos">
                    <option value="bottom" selected>底部置中</option>
                    <option value="top">頂部置中</option>
                  </select>
                </div>
                <div class="form-group">
                  <label>操作</label>
                  <button class="btn-tiny" style="height:35px; background:rgba(99,102,241,0.25); color:#a5b4fc; font-weight:700;" onclick="burnSubtitles();">💾 壓制新影片</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom: Full-Width History Showcase Drawer with Video Previews -->
    <div class="glass-card collapsible open" id="history-drawer">
      <div class="collapsible-header" onclick="toggleDrawer('history-drawer', 'history-arrow')">
        <div style="display:flex; align-items:center; gap:0.75rem;">
          <span style="font-size:1.05rem; font-weight:700; color:var(--text-main);">🕒 最近生成紀錄 (History)</span>
          <span class="badge-status completed" id="history-count-badge">0 部影片</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.75rem;" onclick="event.stopPropagation();">
          <div class="chips-container">
            <span class="chip active" id="view-grid" onclick="setHistoryView('grid')">🖼️ 網格模式</span>
            <span class="chip" id="view-list" onclick="setHistoryView('list')">📋 列表模式</span>
            <button class="btn-tiny" id="btn-unhide" style="display:none;" onclick="resetHiddenHistory()">👁️ 顯示已隱藏 (<span id="hidden-count">0</span>)</button>
          </div>
          <span id="history-arrow" style="cursor:pointer; color:var(--text-muted);">▲</span>
        </div>
      </div>
      <div class="collapsible-content" style="padding:1rem 0 0 0; border:none; display:flex;">
        <div id="history-container" class="history-grid">
          <span style="color:var(--text-muted); font-size:0.85rem;">目前尚無歷史紀錄。</span>
        </div>
      </div>
    </div>
  </main>

  <div id="toast" class="toast"></div>

  <script>
    let currentResult = null;
    let currentStartImagePath = null;
    let currentRefSubjects = [];
    let currentVideoMode = 'text';
    let historyViewMode = localStorage.getItem('minimax_history_view') || 'grid';
    let hiddenHistoryKeys = JSON.parse(localStorage.getItem('minimax_hidden_history') || '[]');
    let serverPaths = {};

    function showToast(msg, duration = 3000) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.style.display = 'block';
      setTimeout(() => { toast.style.display = 'none'; }, duration);
    }

    function selectPathPreset(presetKey) {
      document.querySelectorAll('#path-presets .chip').forEach(c => c.classList.remove('active'));
      const activeChip = document.getElementById('chip-' + presetKey);
      if (activeChip) activeChip.classList.add('active');
      const targetPath = serverPaths[presetKey] || '';
      if (targetPath) {
        document.getElementById('output-dir').value = targetPath;
        localStorage.setItem('minimax_output_dir', targetPath);
      }
    }

    function saveOutputDir(val) {
      localStorage.setItem('minimax_output_dir', val.trim());
      document.querySelectorAll('#path-presets .chip').forEach(c => c.classList.remove('active'));
      if (val === serverPaths.default) document.getElementById('chip-default')?.classList.add('active');
      else if (val === serverPaths.desktop) document.getElementById('chip-desktop')?.classList.add('active');
      else if (val === serverPaths.downloads) document.getElementById('chip-downloads')?.classList.add('active');
      else if (val === serverPaths.movies) document.getElementById('chip-movies')?.classList.add('active');
    }

    function openCurrentFolder() {
      const dir = (document.getElementById('output-dir')?.value || '').trim();
      fetch('/api/open-folder', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dir_path: dir})
      });
    }

    function toggleCollapsible(el) {
      el.parentElement.classList.toggle('open');
    }

    function toggleDrawer(drawerId, arrowId) {
      const drawer = document.getElementById(drawerId);
      if (!drawer) return;
      drawer.classList.toggle('open');
      const arrow = document.getElementById(arrowId);
      if (arrow) arrow.innerText = drawer.classList.contains('open') ? '▲' : '▼';
    }

    function switchVideoMode(mode) {
      currentVideoMode = mode;
      document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
      const activeTab = document.getElementById(`tab-mode-${mode}`);
      if (activeTab) activeTab.classList.add('active');

      document.getElementById('start-image-panel').style.display = (mode === 'image') ? 'flex' : 'none';
      document.getElementById('reference-panel').style.display = (mode === 'reference') ? 'flex' : 'none';
    }

    function handleProfileChange(val) {
      const longBox = document.getElementById('long-mode-box');
      if (longBox) longBox.style.display = (val === 'long') ? 'flex' : 'none';

      if (val === 'fast') {
        document.getElementById('custom-w').value = 768;
        document.getElementById('custom-h').value = 448;
        document.getElementById('custom-steps').value = 10;
      } else if (val === 'standard') {
        document.getElementById('custom-w').value = 960;
        document.getElementById('custom-h').value = 544;
        document.getElementById('custom-steps').value = 15;
      } else if (val === 'hd720') {
        document.getElementById('custom-w').value = 1280;
        document.getElementById('custom-h').value = 720;
        document.getElementById('custom-steps').value = 15;
      } else if (val === 'long') {
        document.getElementById('custom-w').value = 768;
        document.getElementById('custom-h').value = 448;
        document.getElementById('custom-steps').value = 10;
      }
    }

    // Start Image Handler
    function setStartImage(filePath, localBlobUrl) {
      if (!filePath && !localBlobUrl) return;
      currentStartImagePath = filePath;
      switchVideoMode('image');
      const thumb = document.getElementById('start-image-thumb');
      const placeholder = document.getElementById('start-image-placeholder');
      const clearBtn = document.getElementById('btn-clear-start-image');
      thumb.src = localBlobUrl || `/api/video-stream?path=${encodeURIComponent(filePath)}`;
      thumb.style.display = 'block';
      placeholder.style.display = 'none';
      clearBtn.style.display = 'block';
      if (filePath) {
        showToast('🎬 已成功設定為影片起始幀 (I2V)！');
      }
    }

    function clearStartImage() {
      currentStartImagePath = null;
      document.getElementById('start-image-thumb').style.display = 'none';
      document.getElementById('start-image-placeholder').style.display = 'block';
      document.getElementById('btn-clear-start-image').style.display = 'none';
    }

    async function handleStartImageUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      const blobUrl = URL.createObjectURL(file);
      setStartImage('', blobUrl);
      showToast('⏳ 正在上傳並儲存圖片...');
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/api/assets/upload', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '上傳失敗');
        currentStartImagePath = data.path;
        showToast('🎬 起始幀已成功就緒！');
      } catch (e) {
        showToast(`❌ 上傳失敗: ${e.message}`);
        clearStartImage();
      }
    }

    // Reference Subjects Handler
    function renderRefSubjectsList() {
      const listEl = document.getElementById('ref-images-list');
      if (!listEl) return;
      if (currentRefSubjects.length === 0) {
        listEl.innerHTML = '<span style="font-size:0.75rem; color:var(--text-muted);">尚無特徵參考圖，可點擊「➕ 加圖」或從下方圖片庫點「🏷️ 設參考」。</span>';
        return;
      }
      listEl.innerHTML = currentRefSubjects.map((ref, idx) => `
        <div style="display:flex; align-items:center; gap:0.4rem; background:rgba(255,255,255,0.06); border:1px solid var(--card-border); padding:0.25rem 0.5rem; border-radius:6px;">
          <img src="/api/video-stream?path=${encodeURIComponent(ref.image || ref.path || '')}" style="width:24px; height:24px; object-fit:cover; border-radius:4px;" />
          <span style="font-size:0.75rem; color:var(--text-main);">${ref.name || 'Subject ' + (idx+1)}</span>
          <button class="btn-tiny" onclick="removeRefSubject(${idx})" style="padding:0.1rem 0.3rem;">✕</button>
        </div>
      `).join('');
    }

    function addRefSubjectImageDirect(filePath, name) {
      if (!filePath) return;
      currentRefSubjects.push({ image: filePath, name: name || 'Subject' });
      switchVideoMode('reference');
      renderRefSubjectsList();
      showToast('🏷️ 已將圖片加入特徵參考庫 (Ref2V)！');
    }

    function removeRefSubject(idx) {
      currentRefSubjects.splice(idx, 1);
      renderRefSubjectsList();
    }

    function addRefSubjectImage() {
      const nameInput = document.getElementById('ref-subject-name');
      const name = (nameInput?.value || '').trim() || 'Subject';
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        showToast('⏳ 正在上傳參考圖...');
        const formData = new FormData();
        formData.append('file', file);
        try {
          const res = await fetch('/api/assets/upload', { method: 'POST', body: formData });
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || '上傳失敗');
          addRefSubjectImageDirect(data.path, name);
          if (nameInput) nameInput.value = '';
        } catch (err) {
          showToast(`❌ ${err.message}`);
        }
      };
      input.click();
    }

    // Image Generation Controls & API
    let currentImageQuality = 'high';

    function setImageQuality(q) {
      currentImageQuality = q;
      document.querySelectorAll('#img-quality-selector .chip').forEach(c => c.classList.remove('active'));
      const activeChip = document.getElementById('qchip-' + q);
      if (activeChip) activeChip.classList.add('active');
    }

    async function loadAvailableImageModels() {
      try {
        const res = await fetch('/api/image/models');
        if (!res.ok) return;
        const data = await res.json();
        const select = document.getElementById('image-model-select');
        if (select && data.models) {
          select.innerHTML = data.models.map(m => `
            <option value="${m.id}" ${m.id === data.default_model ? 'selected' : ''}>${m.display_name}</option>
          `).join('');
        }
      } catch (e) {}
    }

    async function saveHFTokenClick() {
      const token = (document.getElementById('hf-token-input')?.value || '').trim();
      if (!token) {
        showToast('⚠️ 請輸入 Hugging Face Token');
        return;
      }
      showToast('⏳ 正在儲存 Hugging Face Token...');
      try {
        const res = await fetch('/api/settings/hf-token', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ token: token })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '儲存失敗');
        showToast('✅ Hugging Face Token 已成功儲存！');
        await loadAvailableImageModels();
      } catch (e) {
        showToast(`❌ ${e.message}`);
      }
    }

    let extremeQualityMode = false;

    function toggleExtremeQuality() {
      extremeQualityMode = !extremeQualityMode;
      const btn = document.getElementById('btn-extreme-quality');
      if (btn) {
        if (extremeQualityMode) {
          btn.innerText = '🔥 極限品質: ON';
          btn.style.background = 'linear-gradient(135deg, rgba(239,68,68,0.3) 0%, rgba(245,158,11,0.3) 100%)';
          btn.style.borderColor = '#ef4444';
          btn.style.color = '#fff';
          showToast('🔥 已開啟極限品質模式（不計生成時間，榨出最高畫質）');
        } else {
          btn.innerText = '🔥 極限品質: OFF';
          btn.style.background = 'rgba(239,68,68,0.08)';
          btn.style.borderColor = 'rgba(239,68,68,0.3)';
          btn.style.color = '#fca5a5';
          showToast('已關閉極限品質模式（恢復標準日常品質）');
        }
      }
    }

    async function handleGenerateImageClick() {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) {
        showToast('⚠️ 請先輸入提示詞！');
        return;
      }
      const model_name = document.getElementById('image-model-select')?.value || 'fhdr-uncensored';
      const output_dir = (document.getElementById('output-dir')?.value || '').trim();
      const modeText = extremeQualityMode ? `${currentImageQuality.toUpperCase()} · 🔥極限` : currentImageQuality.toUpperCase();
      showToast(`🎨 開始使用 ${model_name.toUpperCase()} (${modeText}) 生成圖片...`);
      try {
        const res = await fetch('/api/image/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            prompt: prompt,
            model_name: model_name,
            quality_profile: currentImageQuality,
            extreme_quality_mode: extremeQualityMode,
            width: parseInt(document.getElementById('custom-w')?.value || 768),
            height: parseInt(document.getElementById('custom-h')?.value || 768),
            steps: parseInt(document.getElementById('custom-steps')?.value || 4),
            quantize: 4,
            count: 1,
            output_dir: output_dir
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '圖片生成失敗');
        showToast('✅ 圖片生成成功！');
        await fetchImageHistory();
      } catch (e) {
        showToast(`❌ ${e.message}`);
      }
    }

    async function fetchImageHistory() {
      try {
        const res = await fetch('/api/image/history');
        const items = await res.json();
        const grid = document.getElementById('images-grid');
        if (!items || items.length === 0) {
          grid.innerHTML = '<span style="font-size:0.75rem; color:var(--text-muted);">尚無生成圖片。點擊上方「🖼️ 生成圖片」即可快速構圖！</span>';
          return;
        }
        grid.innerHTML = items.map(img => {
          const filePath = img.output_path || img.path || '';
          const safePath = filePath.replace(/'/g, "\\'");
          return `
          <div class="img-card">
            <img src="/api/video-stream?path=${encodeURIComponent(filePath)}" alt="${img.prompt || ''}" onclick="setStartImage('${safePath}');" title="點擊直接設為影片起始幀 (I2V)" />
            <div class="img-card-actions">
              <div style="font-size:0.68rem; color:var(--text-main); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:0.1rem 0.2rem;" title="${img.prompt || ''}">${img.prompt || '無描述'}</div>
              <div style="display:flex; gap:0.25rem;">
                <button class="btn-card-mini" style="flex:1; background:rgba(99,102,241,0.3); color:#a5b4fc; font-weight:700;" onclick="setStartImage('${safePath}');" title="設為影片起始幀 (I2V)">🎬 設起始幀</button>
                <button class="btn-card-mini" onclick="addRefSubjectImageDirect('${safePath}', '${(img.prompt||'Subject').slice(0,10).replace(/'/g, "\\'")}')" title="加入特徵參考庫 (Ref2V)">🏷️ 參考</button>
                <button class="btn-card-mini" onclick="openOutputFolder('${safePath}')" title="在 Finder 中定位檔案">📂</button>
              </div>
            </div>
          </div>
        `}).join('');
      } catch (e) {}
    }

    // Video Generation
    async function handleGenerateVideoClick() {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) {
        showToast('⚠️ 請輸入提示詞');
        return;
      }
      const profile = document.getElementById('video-profile').value;
      const isLong = (profile === 'long');
      const targetDuration = isLong ? parseFloat(document.getElementById('long-target-duration').value || '10.0') : null;
      const output_dir = (document.getElementById('output-dir')?.value || '').trim();

      const body = {
        prompt: prompt,
        profile: profile,
        mode: currentVideoMode,
        width: parseInt(document.getElementById('custom-w').value),
        height: parseInt(document.getElementById('custom-h').value),
        duration_sec: isLong ? 2.0 : 2.0,
        steps: parseInt(document.getElementById('custom-steps').value),
        start_image: (currentVideoMode === 'image') ? currentStartImagePath : null,
        references: (currentVideoMode === 'reference') ? currentRefSubjects : null,
        long_mode: isLong,
        target_duration_sec: targetDuration,
        output_dir: output_dir
      };

      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '生成啟動失敗');
        showToast('🚀 已啟動生成！');
        syncJobState();
      } catch (e) {
        showToast(`❌ ${e.message}`);
      }
    }

    // Queue Management
    async function handleAddToQueueClick() {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) {
        showToast('⚠️ 請輸入提示詞');
        return;
      }
      const profile = document.getElementById('video-profile').value;
      const isLong = (profile === 'long');
      const targetDuration = isLong ? parseFloat(document.getElementById('long-target-duration').value || '10.0') : null;
      const output_dir = (document.getElementById('output-dir')?.value || '').trim();

      try {
        const res = await fetch('/api/queue/batch-add', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            prompts_text: prompt,
            profile: profile,
            width: parseInt(document.getElementById('custom-w').value),
            height: parseInt(document.getElementById('custom-h').value),
            duration_sec: isLong ? 2.0 : 2.0,
            steps: parseInt(document.getElementById('custom-steps').value),
            mode: currentVideoMode,
            start_image: (currentVideoMode === 'image') ? currentStartImagePath : null,
            references: (currentVideoMode === 'reference') ? currentRefSubjects : null,
            output_dir: output_dir
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '加入排程失敗');
        showToast('➕ 已加入排程！');
        document.getElementById('prompt').value = '';
        fetchQueue();
      } catch (e) {
        showToast(`❌ ${e.message}`);
      }
    }

    async function fetchQueue() {
      try {
        const res = await fetch('/api/queue');
        const data = await res.json();
        const list = document.getElementById('queue-list');
        const badge = document.getElementById('queue-count-badge');
        const autoCheck = document.getElementById('auto-queue-toggle');
        if (autoCheck) autoCheck.checked = data.auto_enabled;
        badge.innerText = `${data.items.length} 筆`;

        if (!data.items || data.items.length === 0) {
          list.innerHTML = '<span style="font-size:0.75rem; color:var(--text-muted);">目前無排程工作。</span>';
          return;
        }

        list.innerHTML = data.items.map(item => `
          <div class="queue-card ${item.status}">
            <div class="queue-header">
              <span class="badge-status ${item.status}">${item.status.toUpperCase()}</span>
              <div style="display:flex; gap:0.3rem;">
                ${item.status === 'completed' && item.output_path ? `
                  <button class="btn-tiny" onclick="event.stopPropagation(); loadHistoryItem({output_path: '${item.output_path}', prompt: '${item.prompt.replace(/'/g, "\\'")}', width:${item.width||768}, height:${item.height||448}, duration_sec:${item.duration_sec||2.0}})">▶️ 播放</button>
                  <button class="btn-tiny" onclick="event.stopPropagation(); openOutputFolder('${item.output_path}')">📂 資料夾</button>
                ` : ''}
                ${item.status === 'failed' || item.status === 'cancelled' ? `<button class="btn-tiny" onclick="queueAction('${item.id}', 'retry')">🔄 重試</button>` : ''}
                <button class="btn-tiny" onclick="queueAction('${item.id}', 'delete')">🗑️</button>
              </div>
            </div>
            <div style="font-size:0.8rem; color:var(--text-main);">${item.prompt}</div>
            ${item.error_message ? `<div style="font-size:0.7rem; color:#f87171;">${item.error_message}</div>` : ''}
          </div>
        `).join('');
      } catch (e) {}
    }

    async function queueAction(id, action) {
      await fetch('/api/queue/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({item_id: id, action: action})
      });
      fetchQueue();
    }

    async function toggleAutoQueue() {
      await fetch('/api/queue/toggle-auto', {method: 'POST'});
      fetchQueue();
    }

    async function cancelCurrentTask() {
      await fetch('/api/generate/cancel', {method: 'POST'});
      showToast('🛑 正在中斷任務...');
    }

    // Player & Subtitles
    function displayResult(res) {
      if (!res) return;
      currentResult = res;
      const player = document.getElementById('video-player');
      const empty = document.getElementById('empty-state');
      const meta = document.getElementById('meta-grid');

      const vPath = res.output_path || (res.output_filename ? `/outputs/${res.output_filename}` : '');
      if (vPath) {
        player.src = `/api/video-stream?path=${encodeURIComponent(vPath)}`;
        player.style.display = 'block';
        empty.style.display = 'none';
        meta.style.display = 'grid';

        if (res.width && res.height) document.getElementById('meta-res').innerText = `${res.width}x${res.height}`;
        if (res.duration_sec) document.getElementById('meta-dur').innerText = `${res.duration_sec}s`;
        if (res.seed !== undefined) document.getElementById('meta-seed').innerText = `${res.seed}`;
        if (res.execution_time_sec) document.getElementById('meta-time').innerText = `${(res.execution_time_sec/60).toFixed(1)}m`;
      }
    }

    function toggleLoop() {
      const p = document.getElementById('video-player');
      p.loop = !p.loop;
      document.getElementById('btn-loop').style.borderColor = p.loop ? '#6366f1' : 'var(--card-border)';
    }

    function setPlaySpeed(val) {
      document.getElementById('video-player').playbackRate = parseFloat(val);
    }

    async function burnSubtitles() {
      const text = document.getElementById('sub-text').value.trim();
      if (!text || !currentResult || !currentResult.output_path) {
        showToast('⚠️ 請先輸入字幕並確認已有載入影片！');
        return;
      }
      showToast('⏳ 正在壓制字幕...');
      try {
        const res = await fetch('/api/subtitles/burn', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            video_path: currentResult.output_path,
            text: text,
            style: document.getElementById('sub-style').value,
            position: document.getElementById('sub-pos').value,
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '字幕壓制失敗');
        showToast('✅ 字幕壓制完成！已儲存新影片！');
        displayResult(data);
        await fetchHistory();
      } catch (e) {
        showToast(`❌ ${e.message}`);
      }
    }

    /* History Management: Grid/List View & Hide Individual Items */
    function setHistoryView(mode) {
      historyViewMode = mode;
      localStorage.setItem('minimax_history_view', mode);
      const gridChip = document.getElementById('view-grid');
      const listChip = document.getElementById('view-list');
      if (gridChip) gridChip.classList.toggle('active', mode === 'grid');
      if (listChip) listChip.classList.toggle('active', mode === 'list');
      fetchHistory();
    }

    function hideHistoryItem(key, event) {
      if (event) event.stopPropagation();
      if (!hiddenHistoryKeys.includes(key)) {
        hiddenHistoryKeys.push(key);
        localStorage.setItem('minimax_hidden_history', JSON.stringify(hiddenHistoryKeys));
      }
      showToast("已隱藏該部影片紀錄", 1500);
      fetchHistory();
    }

    function resetHiddenHistory() {
      hiddenHistoryKeys = [];
      localStorage.setItem('minimax_hidden_history', JSON.stringify([]));
      showToast("已重置並顯示所有隱藏影片", 1500);
      fetchHistory();
    }

    async function fetchHistory() {
      try {
        const res = await fetch('/api/history');
        if (!res.ok) return [];
        const items = await res.json();
        const container = document.getElementById('history-container');
        const badge = document.getElementById('history-count-badge');
        const unhideBtn = document.getElementById('btn-unhide');
        const hiddenCountEl = document.getElementById('hidden-count');

        if (unhideBtn && hiddenCountEl) {
          const hiddenCount = hiddenHistoryKeys.length;
          hiddenCountEl.innerText = hiddenCount;
          unhideBtn.style.display = hiddenCount > 0 ? 'inline-block' : 'none';
        }

        if (!items || items.length === 0) {
          badge.innerText = '0 部影片';
          container.innerHTML = '<span style="color:var(--text-muted); font-size:0.85rem;">目前尚無歷史紀錄。</span>';
          return [];
        }

        const visibleItems = items.filter(item => {
          const key = item.output_filename || item.output_path;
          return !hiddenHistoryKeys.includes(key);
        });

        badge.innerText = `${visibleItems.length} 部影片`;

        if (visibleItems.length === 0) {
          container.innerHTML = '<span style="color:var(--text-muted); font-size:0.85rem;">所有歷史影片均已隱藏，點擊右上角可還原顯示。</span>';
          return visibleItems;
        }

        if (historyViewMode === 'list') {
          container.className = 'history-list-view';
          container.innerHTML = visibleItems.map(item => {
            const key = item.output_filename || item.output_path;
            const videoSrc = item.output_path ? `/api/video-stream?path=${encodeURIComponent(item.output_path)}` : (item.output_filename ? `/outputs/${item.output_filename}` : '');
            return `
              <div class="history-list-row" onclick='loadHistoryItem(${JSON.stringify(item)})'>
                <div style="display:flex; align-items:center; gap:0.75rem; min-width:0; flex:1;">
                  <span style="font-size:1.2rem; color:#818cf8;">🎬</span>
                  <div style="min-width:0; flex:1;">
                    <div style="font-size:0.85rem; font-weight:600; color:var(--text-main); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                      ${item.prompt}
                    </div>
                    <div style="font-size:0.7rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace; display:flex; gap:0.5rem; margin-top:0.2rem;">
                      <span>${item.output_filename || ''}</span>
                      <span>•</span>
                      <span>${item.width}x${item.height}</span>
                      <span>•</span>
                      <span>${item.duration_sec}s</span>
                      ${item.execution_time_sec ? `<span>• ${(item.execution_time_sec/60).toFixed(1)}m</span>` : ''}
                    </div>
                  </div>
                </div>
                <div style="display:flex; gap:0.4rem; align-items:center;" onclick="event.stopPropagation();">
                  <button class="btn-tiny" onclick='loadHistoryItem(${JSON.stringify(item)})' title="在播放器播放">▶️ 播放</button>
                  <button class="btn-tiny" onclick="openOutputFolder('${item.output_path || ''}')" title="在 Finder 開啟">📂 資料夾</button>
                  <button class="btn-tiny" onclick="hideHistoryItem('${key}', event)" title="從列表中隱藏">✕ 隱藏</button>
                </div>
              </div>
            `;
          }).join('');
        } else {
          container.className = 'history-grid';
          container.innerHTML = visibleItems.map(item => {
            const key = item.output_filename || item.output_path;
            const videoSrc = item.output_path ? `/api/video-stream?path=${encodeURIComponent(item.output_path)}` : (item.output_filename ? `/outputs/${item.output_filename}` : '');
            return `
              <div class="history-item" onclick='loadHistoryItem(${JSON.stringify(item)})'>
                <button class="btn-hide-item" onclick="hideHistoryItem('${key}', event)" title="隱藏這部影片">✕</button>
                <div class="history-video-thumb">
                  ${videoSrc ? `<video src="${videoSrc}" preload="metadata" muted onmouseover="this.play()" onmouseout="this.pause()"></video>` : '<div style="color:var(--text-muted); font-size:0.75rem;">無預覽</div>'}
                </div>
                <div class="history-prompt" title="${item.prompt}">${item.prompt}</div>
                <div class="history-footer">
                  <span>${item.width}x${item.height} (${item.duration_sec}s)</span>
                  <span>${item.execution_time_sec ? (item.execution_time_sec/60).toFixed(1)+'m' : ''}</span>
                </div>
              </div>
            `;
          }).join('');
        }
        return visibleItems;
      } catch (e) { return []; }
    }

    function loadHistoryItem(item) {
      if (!item) return;
      displayResult(item);
    }

    function openOutputFolder(p) {
      fetch('/api/open-folder', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({file_path: p})
      });
    }

    let lastHandledResultPath = null;

    // Reactive State Sync Loop
    async function syncJobState() {
      try {
        const res = await fetch('/api/job');
        const data = await res.json();
        const pCard = document.getElementById('progress-card');
        const pFill = document.getElementById('progress-fill');
        const pStage = document.getElementById('progress-stage-text');
        const pPct = document.getElementById('progress-pct-text');
        const pTime = document.getElementById('progress-time-text');
        const stateText = document.getElementById('system-state-text');

        if (data.is_running) {
          pCard.style.display = 'flex';
          pFill.style.width = `${Math.max(5, Math.round(data.progress * 100))}%`;
          pStage.innerText = data.stage || '生成中...';
          pPct.innerText = `${Math.round(data.progress * 100)}%`;
          if (pTime) {
            const min = (data.elapsed_sec / 60).toFixed(1);
            pTime.innerText = `⏱️ ${min}m`;
          }
          stateText.innerText = (data.job_type === 'IMAGE' ? '🎨 圖片生成中' : '🎬 影片去噪中');
        } else {
          pCard.style.display = 'none';
          stateText.innerText = 'Ready';
          if (data.result && data.result.output_path && data.result.output_path !== lastHandledResultPath) {
            lastHandledResultPath = data.result.output_path;
            displayResult(data.result);
            fetchHistory();
            fetchQueue();
          }
        }
      } catch (e) {}
    }

    async function syncStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.memory) {
          const used = data.memory.active_gib !== undefined && data.memory.active_gib > 0 ? data.memory.active_gib : (data.memory.total_ram_gib * (100 - data.memory.free_pct) / 100).toFixed(1);
          const total = data.memory.total_ram_gib || 32.0;
          const memUsedEl = document.getElementById('mem-used');
          const memTotalEl = document.getElementById('mem-total');
          if (memUsedEl) memUsedEl.innerText = `${used} GB`;
          if (memTotalEl) memTotalEl.innerText = `${total} GB`;
        }
        if (data.default_output_dir && !serverPaths.default) {
          serverPaths.default = data.default_output_dir;
          serverPaths.desktop = data.desktop_dir;
          serverPaths.downloads = data.downloads_dir;
          serverPaths.movies = data.movies_dir;

          const outInput = document.getElementById('output-dir');
          if (outInput) {
            const saved = localStorage.getItem('minimax_output_dir');
            if (saved) {
              outInput.value = saved;
              saveOutputDir(saved);
            } else {
              outInput.value = data.default_output_dir;
              saveOutputDir(data.default_output_dir);
            }
          }
        }
      } catch (e) {}
    }

    // Init
    async function initApp() {
      await loadAvailableImageModels();
      await syncStatus();
      await syncJobState();
      await fetchQueue();
      await fetchImageHistory();
      const historyItems = await fetchHistory();
      if (historyItems && historyItems.length > 0 && !currentResult) {
        displayResult(historyItems[0]);
      }
      setInterval(syncJobState, 1500);
      setInterval(fetchQueue, 2500);
      setInterval(syncStatus, 4000);
    }

    window.addEventListener('DOMContentLoaded', initApp);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


if __name__ == "__main__":
    port = 7860
    print(f"🎬 啟動 色色 Studio (http://127.0.0.1:{port}/)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
