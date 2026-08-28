#!/usr/bin/env python3
"""Lightweight local Web UI for MiniMax-H3 on Apple Silicon with Prompt Queue and Post-Processing Subtitles."""

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
LOGS_DIR = BASE_DIR / "logs"
QUEUE_FILE = LOGS_DIR / "queue.jsonl"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import engine
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="MiniMax-H3 Local Studio")

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
                            # Reset running items to queued on restart
                            if item.get("status") == "running":
                                item["status"] = "queued"
                            PROMPT_QUEUE.append(item)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Warning: Failed to load queue: {e}", file=sys.stderr)


def save_queue_to_file():
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            for item in PROMPT_QUEUE:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to save queue: {e}", file=sys.stderr)


load_queue_from_file()


class GenerateRequest(BaseModel):
    prompt: str
    profile: str = "fast"
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
    # Truncate clean message
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
):
    global CURRENT_JOB
    cancel_event = threading.Event()
    with JOB_LOCK:
        CURRENT_JOB["is_running"] = True
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
        dur = max(0.5, min(15.08, duration_sec))
        st = max(4, min(60, steps))
        seed_val = seed if seed >= 0 else None
        custom_out = output_dir.strip() if output_dir and output_dir.strip() else str(OUTPUTS_DIR)

        res = engine.generate_video(
            prompt=prompt,
            width=w,
            height=h,
            duration_sec=dur,
            seed=seed_val,
            steps=st,
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
        # User manually cancelled
        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["stage"] = "Cancelled."
            CURRENT_JOB["error"] = "Generation was cancelled by user."
            CURRENT_JOB["progress"] = 0.0
            CURRENT_JOB["cancel_event"] = None

        # Move to end of queue with cancelled status
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
        # Generation failed
        err_msg = str(e)
        friendly_err = summarize_error(err_msg)
        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["stage"] = "Generation failed."
            CURRENT_JOB["error"] = friendly_err
            CURRENT_JOB["progress"] = 0.0
            CURRENT_JOB["cancel_event"] = None

        # Move failed item to end of queue with error message
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
        # Find next queued item
        next_item = None
        with QUEUE_LOCK:
            for item in PROMPT_QUEUE:
                if item.get("status") == "queued":
                    next_item = item
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
                },
                daemon=True,
            ).start()


threading.Thread(target=queue_worker_loop, daemon=True).start()


@app.get("/api/status")
async def get_status():
    mem = engine.get_system_memory_status()
    with JOB_LOCK:
        job_info = {
            "is_running": CURRENT_JOB["is_running"],
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
            "stage": CURRENT_JOB["stage"],
            "progress": round(CURRENT_JOB["progress"], 2),
            "elapsed_sec": elapsed,
            "prompt": CURRENT_JOB.get("prompt", ""),
            "result": CURRENT_JOB["result"],
            "error": CURRENT_JOB["error"],
            "active_queue_id": CURRENT_JOB.get("active_queue_id"),
        }


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
        CURRENT_JOB["stage"] = "Cancelling task..."
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
        for line in lines:
            item = {
                "id": str(uuid.uuid4())[:8],
                "prompt": line,
                "profile": req.profile,
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
            PROMPT_QUEUE.append(item)
            added_items.append(item)
        save_queue_to_file()
    return {"status": "ok", "added_count": len(added_items), "items": PROMPT_QUEUE}


@app.post("/api/queue/action")
async def queue_action(req: QueueActionRequest):
    global PROMPT_QUEUE
    with QUEUE_LOCK:
        target_idx = None
        for idx, item in enumerate(PROMPT_QUEUE):
            if item["id"] == req.item_id:
                target_idx = idx
                break
        if target_idx is None:
            raise HTTPException(status_code=404, detail="找不到指定排程項目")

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
    if not v_path.exists() or not v_path.is_file():
        raise HTTPException(status_code=404, detail="找不到指定的影片檔案")

    try:
        out_path = engine.burn_subtitles_to_video(
            input_path=v_path,
            text=req.text.strip(),
            start_sec=req.start_sec,
            end_sec=req.end_sec,
            position=req.position,
            style=req.style,
            font_size=req.font_size,
        )
        return {
            "status": "ok",
            "output_path": out_path,
            "output_filename": Path(out_path).name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"字幕壓制失敗: {e}")


@app.get("/api/history")
async def get_history():
    return engine.get_history(limit=30)


@app.get("/api/video-stream")
async def stream_video(path: str = Query(...)):
    """Serve video directly from any local folder with standard headers."""
    decoded_path = urllib.parse.unquote(path)
    file_path = Path(decoded_path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="影片檔案不存在")
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
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


# Complete SPA HTML with Prompt Queue & Subtitle Editor
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-TW" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MiniMax-H3 Local Studio</title>
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
      padding: 1rem 2rem;
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
    .brand-text h1 { font-size: 1.15rem; font-weight: 700; }
    .brand-text p { font-size: 0.75rem; color: var(--text-muted); }
    .system-status { display: flex; align-items: center; gap: 1rem; font-size: 0.8rem; }
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
      flex: 1; max-width: 1360px; margin: 0 auto;
      padding: 2rem 1.5rem; width: 100%;
      display: grid; grid-template-columns: 1fr 1.1fr; gap: 2rem;
    }
    @media (max-width: 1024px) { main { grid-template-columns: 1fr; } }

    .glass-card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      display: flex; flex-direction: column; gap: 1.25rem;
    }

    .card-title {
      font-size: 1.1rem; font-weight: 700;
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--card-border); padding-bottom: 0.75rem;
    }

    .form-group { display: flex; flex-direction: column; gap: 0.5rem; }
    label { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; }
    textarea, select, input {
      background: rgba(11, 15, 23, 0.7); border: 1px solid var(--card-border);
      color: var(--text-main); padding: 0.75rem 1rem; border-radius: 10px;
      font-family: inherit; font-size: 0.95rem; outline: none; transition: 0.2s ease;
    }
    textarea:focus, select:focus, input:focus {
      border-color: var(--primary-accent); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
    }
    textarea { min-height: 90px; resize: vertical; line-height: 1.5; }

    .chips-container { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .chip {
      background: rgba(255, 255, 255, 0.05); border: 1px solid var(--card-border);
      font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 6px;
      cursor: pointer; color: var(--text-muted); transition: 0.15s ease;
    }
    .chip:hover { background: rgba(99, 102, 241, 0.2); color: var(--text-main); border-color: var(--primary-accent); }
    .chip.active { background: rgba(99, 102, 241, 0.35); color: #c7d2fe; border-color: #818cf8; font-weight: 600; }

    .duration-control-box, .path-control-box {
      background: rgba(11, 15, 23, 0.5); border: 1px solid var(--card-border);
      border-radius: 10px; padding: 0.85rem 1rem; display: flex; flex-direction: column; gap: 0.6rem;
    }
    .duration-inputs { display: flex; align-items: center; gap: 1rem; }
    .duration-num-input { width: 95px; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.05rem; text-align: center; color: #818cf8; }
    .duration-slider { flex: 1; height: 6px; accent-color: #6366f1; cursor: pointer; }
    .duration-calc-badge { font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; display: flex; justify-content: space-between; }

    .path-row { display: flex; gap: 0.5rem; align-items: center; }
    .path-input { flex: 1; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; padding: 0.6rem 0.8rem; }

    .collapsible {
      border: 1px solid var(--card-border); border-radius: 10px;
      overflow: hidden; background: rgba(11, 15, 23, 0.4);
    }
    .collapsible-header {
      padding: 0.75rem 1rem; cursor: pointer; display: flex;
      justify-content: space-between; align-items: center;
      font-size: 0.85rem; font-weight: 600; color: var(--text-muted);
    }
    .collapsible-content { padding: 1rem; display: none; flex-direction: column; gap: 1rem; border-top: 1px solid var(--card-border); }
    .collapsible.open .collapsible-content { display: flex; }

    /* Button Actions */
    .btn-row { display: flex; gap: 0.75rem; }
    .btn-generate {
      flex: 1; background: var(--primary-gradient); border: none; color: white;
      font-size: 1.05rem; font-weight: 700; padding: 0.95rem; border-radius: 12px;
      cursor: pointer; transition: all 0.25s ease; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
      display: flex; justify-content: center; align-items: center; gap: 0.5rem;
    }
    .btn-generate:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6); }
    .btn-cancel {
      flex: 1; background: var(--danger-gradient); border: none; color: white;
      font-size: 1.05rem; font-weight: 700; padding: 0.95rem; border-radius: 12px;
      cursor: pointer; transition: all 0.2s ease; box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
      display: none; justify-content: center; align-items: center; gap: 0.5rem;
    }
    .btn-cancel:hover { box-shadow: 0 6px 24px rgba(239, 68, 68, 0.6); transform: translateY(-2px); }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.06); border: 1px solid var(--card-border);
      color: var(--text-main); padding: 0.6rem 0.85rem; border-radius: 8px;
      font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: 0.2s ease;
      display: flex; align-items: center; justify-content: center; gap: 0.4rem; text-decoration: none;
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.12); border-color: rgba(255, 255, 255, 0.2); }

    .progress-box {
      background: rgba(11, 15, 23, 0.9); border: 1px solid var(--card-border);
      border-radius: 12px; padding: 1rem; display: none; flex-direction: column; gap: 0.75rem;
    }
    .progress-bar-bg { height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; overflow: hidden; }
    .progress-bar-fill { height: 100%; background: var(--primary-gradient); width: 0%; transition: width 0.3s ease; }
    .stage-text { font-size: 0.85rem; color: #93c5fd; font-family: 'JetBrains Mono', monospace; }

    /* Queue UI */
    .queue-list { display: flex; flex-direction: column; gap: 0.6rem; max-height: 320px; overflow-y: auto; padding-right: 0.3rem; }
    .queue-card {
      background: rgba(11, 15, 23, 0.6); border: 1px solid var(--card-border);
      border-radius: 8px; padding: 0.65rem 0.85rem; display: flex; flex-direction: column; gap: 0.4rem;
      transition: 0.2s ease;
    }
    .queue-card.running { border-color: #10b981; background: rgba(16, 185, 129, 0.08); }
    .queue-card.failed { border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.05); }
    .queue-header { display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; }
    .badge-status {
      padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.7rem; text-transform: uppercase;
    }
    .badge-status.queued { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .badge-status.running { background: rgba(16, 185, 129, 0.25); color: #34d399; }
    .badge-status.completed { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; }
    .badge-status.failed { background: rgba(239, 68, 68, 0.25); color: #f87171; }
    .badge-status.cancelled { background: rgba(156, 163, 175, 0.2); color: #9ca3af; }
    .badge-status.paused { background: rgba(107, 114, 128, 0.2); color: #9ca3af; }
    .queue-prompt-text { font-size: 0.85rem; color: var(--text-main); word-break: break-word; line-height: 1.35; }
    .queue-actions { display: flex; gap: 0.4rem; justify-content: flex-end; align-items: center; }
    .btn-tiny {
      background: rgba(255, 255, 255, 0.06); border: 1px solid var(--card-border);
      color: var(--text-main); font-size: 0.7rem; padding: 0.2rem 0.45rem; border-radius: 4px; cursor: pointer;
    }
    .btn-tiny:hover { background: rgba(255, 255, 255, 0.15); }

    /* Player and Result */
    .player-container {
      border-radius: 12px; overflow: hidden; background: #000;
      border: 1px solid var(--card-border); aspect-ratio: 16 / 9;
      display: flex; align-items: center; justify-content: center; position: relative;
    }
    video { width: 100%; height: 100%; object-fit: contain; }
    .empty-state { color: var(--text-muted); font-size: 0.9rem; text-align: center; padding: 2rem; }

    .player-control-bar {
      display: flex; justify-content: space-between; align-items: center;
      background: rgba(11, 15, 23, 0.7); border: 1px solid var(--card-border);
      border-radius: 10px; padding: 0.5rem 0.85rem; font-size: 0.8rem;
    }
    .play-mode-chips { display: flex; gap: 0.35rem; align-items: center; }
    .play-chip {
      background: rgba(255, 255, 255, 0.05); border: 1px solid var(--card-border);
      padding: 0.25rem 0.55rem; border-radius: 6px; cursor: pointer; font-size: 0.75rem; color: var(--text-muted);
    }
    .play-chip.active { background: rgba(99, 102, 241, 0.35); color: #c7d2fe; border-color: #818cf8; font-weight: 600; }

    .metadata-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
    .meta-card {
      background: rgba(11, 15, 23, 0.6); border: 1px solid var(--card-border);
      padding: 0.6rem; border-radius: 8px; text-align: center;
    }
    .meta-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .meta-val { font-size: 0.95rem; font-weight: 700; color: var(--text-main); font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem; }

    /* History section */
    .history-section { grid-column: 1 / -1; margin-top: 1rem; }
    .history-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1rem; max-height: 480px; overflow-y: auto; padding-right: 0.5rem;
    }
    .history-item {
      background: rgba(11, 15, 23, 0.6); border: 1px solid var(--card-border);
      border-radius: 10px; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem;
      cursor: pointer; transition: all 0.2s ease;
    }
    .history-item:hover { border-color: var(--primary-accent); transform: translateY(-2px); }
    .history-video-thumb {
      width: 100%; height: 140px; background: #000; border-radius: 6px;
      overflow: hidden; display: flex; align-items: center; justify-content: center;
    }
    .history-prompt { font-size: 0.8rem; color: var(--text-main); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.3; }
    .history-footer { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

    .toast-banner {
      position: fixed; bottom: 20px; right: 20px;
      background: rgba(30, 41, 59, 0.95); border: 1px solid var(--card-border);
      color: white; padding: 0.75rem 1.25rem; border-radius: 10px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4); display: none; z-index: 100; font-size: 0.9rem; font-weight: 600;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-badge">H3</div>
      <div class="brand-text">
        <h1>MiniMax-H3 Local Studio</h1>
        <p>Apple Silicon MLX 8-bit Native Inference</p>
      </div>
    </div>
    <div class="system-status">
      <div class="status-pill">
        <span class="indicator-dot" id="pressure-dot"></span>
        <span id="pressure-text">RAM: Checking...</span>
      </div>
      <div class="status-pill" id="swap-pill">
        <span>Swap: 0.0 GB</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Left: Unified Generation Controls & Prompt Queue -->
    <div class="glass-card">
      <div class="card-title">
        <span>✨ 提示詞與排程生成 (Prompt & Queue)</span>
        <label style="display:flex; align-items:center; gap:0.4rem; cursor:pointer; font-size:0.8rem; font-weight:normal; color:var(--text-muted);">
          <input type="checkbox" id="auto-queue-chk" onchange="toggleAutoQueue(this.checked)" checked>
          <span>🔁 自動連續生成</span>
        </label>
      </div>

      <!-- Single Unified Prompt Input for both Single & Batch -->
      <div class="form-group">
        <label for="prompt">Prompt (提示詞 / 支援單行或多行批次輸入)</label>
        <textarea id="prompt" placeholder="可輸入單筆提示詞立即生成，亦可一次貼入多行提示詞（一行一筆）進行批次排程...&#10;例：&#10;A corgi running through a vibrant grassy field, golden hour lighting&#10;Cyberpunk city street at night, neon reflections in puddles&#10;A cute red panda eating bamboo leaves in misty mountain forest" style="min-height:105px;"></textarea>
        <div class="chips-container">
          <span class="chip" onclick="setPrompt(this.innerText)">A golden retriever catching a frisbee on the beach, splashing ocean waves</span>
          <span class="chip" onclick="setPrompt(this.innerText)">Cyberpunk city street at night, neon reflections in puddles, flying cars</span>
          <span class="chip" onclick="setPrompt(this.innerText)">A cute red panda eating bamboo leaves in misty mountain forest</span>
        </div>
      </div>

      <div class="form-group">
        <label for="profile">Generation Profile</label>
        <select id="profile" onchange="onProfileChange()">
          <option value="fast" selected>⚡️ 快速測試 (768×448 / ~2s / 10 steps)</option>
          <option value="standard">🎬 540p 標準 (960×544 / ~2.5s / 15 steps)</option>
          <option value="720p_short">🌟 720p Short (1280×720 / ~2s / 15 steps)</option>
          <option value="720p">💎 720p 標準 (1280×720 / ~3s / 20 steps)</option>
          <option value="custom">⚙️ 自訂參數 (Custom)</option>
        </select>
      </div>

      <!-- Duration Control -->
      <div class="form-group">
        <label>影片長度 (支援手動輸入秒數，上限 15.0 秒)</label>
        <div class="duration-control-box">
          <div class="duration-inputs">
            <input type="number" id="duration-num" min="0.5" max="15.0" step="0.1" value="2.0" class="duration-num-input" oninput="onDurationNumInput(this.value)">
            <span style="font-weight:600; color:var(--text-muted);">秒 (s)</span>
            <input type="range" id="duration-slider" min="0.5" max="15.0" step="0.1" value="2.0" class="duration-slider" oninput="onDurationSliderInput(this.value)">
          </div>
          <div class="duration-calc-badge">
            <span id="duration-frames-text">換算幀數: 56 幀 (約 2.33 秒)</span>
            <span>建議安全範圍: 2.0s ~ 3.0s</span>
          </div>
        </div>
      </div>

      <!-- Output Directory Selector -->
      <div class="form-group">
        <label>📁 影片產生位置 (Output Folder)</label>
        <div class="path-control-box">
          <div class="path-row">
            <input type="text" id="output-dir" class="path-input" placeholder="/Users/.../outputs" onchange="saveOutputDir(this.value)">
            <button class="btn-secondary" style="flex:0 0 auto; padding:0.55rem 0.85rem;" onclick="openCurrentFolder()" title="在 Finder 中開啟">📂 開啟</button>
          </div>
          <div class="chips-container" id="path-presets">
            <span class="chip active" id="chip-default" onclick="selectPathPreset('default')">📁 專案預設</span>
            <span class="chip" id="chip-desktop" onclick="selectPathPreset('desktop')">🖥️ 桌面 (Desktop)</span>
            <span class="chip" id="chip-downloads" onclick="selectPathPreset('downloads')">📥 下載 (Downloads)</span>
            <span class="chip" id="chip-movies" onclick="selectPathPreset('movies')">🎬 影片 (Movies)</span>
          </div>
        </div>
      </div>

      <!-- Advanced Drawer -->
      <div class="collapsible" id="advanced-drawer">
        <div class="collapsible-header" onclick="toggleDrawer('advanced-drawer', 'adv-arrow')">
          <span>🛠️ 進階解析度與步數 (Advanced Settings)</span>
          <span id="adv-arrow">▼</span>
        </div>
        <div class="collapsible-content">
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
            <div class="form-group">
              <label for="width">寬度 (Width)</label>
              <input type="number" id="width" value="768" step="16" min="128" max="1344">
            </div>
            <div class="form-group">
              <label for="height">高度 (Height)</label>
              <input type="number" id="height" value="448" step="16" min="128" max="1344">
            </div>
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
            <div class="form-group">
              <label for="steps">Sampling Steps</label>
              <input type="number" id="steps" value="10" min="4" max="50">
            </div>
            <div class="form-group">
              <label for="seed">Seed (-1 為隨機)</label>
              <input type="number" id="seed" value="-1">
            </div>
          </div>
        </div>
      </div>

      <!-- Unified Action Buttons -->
      <div class="btn-row">
        <button class="btn-generate" id="btn-gen" onclick="handleGenerateClick()">
          <span>🚀 立即生成 (Generate)</span>
        </button>
        <button class="btn-secondary" id="btn-add-single-queue" onclick="handleAddToQueueClick()" title="將上方輸入的提示詞（單筆或多筆）加入排程隊列">
          <span>➕ 批次加入排程</span>
        </button>
        <button class="btn-cancel" id="btn-cancel" onclick="cancelCurrentGeneration()">
          <span>🛑 中止生成 (Cancel)</span>
        </button>
      </div>

      <!-- Progress Box -->
      <div class="progress-box" id="progress-box">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">
          <span class="stage-text" id="stage-text">Preparing...</span>
          <div style="display:flex; gap:0.5rem; align-items:center; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">
            <span id="stage-pct" style="color:#818cf8; font-weight:700;">0%</span>
            <span id="time-text" style="color:var(--text-muted);">0s</span>
          </div>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="progress-bar"></div>
        </div>
      </div>

      <!-- Unified Queue List Container -->
      <div class="form-group" style="margin-top:0.5rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
          <span style="font-size:0.85rem; font-weight:700; color:var(--text-muted); display:flex; align-items:center; gap:0.5rem;">
            <span>📋 排程清單 (Queue)</span>
            <span class="badge-status queued" id="queue-count-badge">0 筆待處理</span>
          </span>
          <button class="btn-tiny" onclick="fetchQueue()" title="重新整理排程">🔄 整理</button>
        </div>
        <div class="queue-list" id="queue-items-container">
          <p style="color:var(--text-muted); font-size:0.8rem; padding:0.5rem 0;">目前排程隊列為空。在上方輸入提示詞後點擊「➕ 批次加入排程」即可新增。</p>
        </div>
      </div>
    </div>

    <!-- Right: Result Showcase, Player & Post-Processing Subtitle Editor -->
    <div class="glass-card">
      <div class="card-title">
        <span>🎬 影片播放與後期編輯</span>
      </div>

      <div class="player-container" id="player-wrap">
        <div class="empty-state" id="empty-state">
          點擊「立即生成」或從下方歷史紀錄選擇影片進行播放與字幕編輯
        </div>
        <video id="video-player" controls style="display:none;"></video>
      </div>

      <!-- Player Controls Toolbar -->
      <div class="player-control-bar" id="player-toolbar" style="display:none;">
        <div class="play-mode-chips">
          <span style="color:var(--text-muted); font-size:0.75rem; margin-right:0.25rem;">播放模式:</span>
          <span class="play-chip active" id="mode-single" onclick="setPlayMode('single')">⏸️ 單次播放</span>
          <span class="play-chip" id="mode-loop" onclick="setPlayMode('loop')">🔁 循環播放</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
          <span style="color:var(--text-muted); font-size:0.75rem;">速度:</span>
          <select id="play-speed" style="padding:0.25rem 0.5rem; font-size:0.75rem;" onchange="setPlaybackSpeed(this.value)">
            <option value="0.5">0.5x</option>
            <option value="0.75">0.75x</option>
            <option value="1.0" selected>1.0x (標準)</option>
            <option value="1.25">1.25x</option>
            <option value="1.5">1.5x</option>
            <option value="2.0">2.0x</option>
          </select>
          <button class="btn-secondary" style="padding:0.25rem 0.5rem; font-size:0.75rem;" onclick="replayVideo()" title="重新播放">🔄 重播</button>
        </div>
      </div>

      <!-- Post-Processing Subtitle Editor (Always visible by default) -->
      <div class="collapsible open" id="sub-editor-drawer">
        <div class="collapsible-header" onclick="toggleDrawer('sub-editor-drawer', 'sub-edit-arrow')">
          <span>💬 後期字幕編輯 (Subtitle & Caption Editor)</span>
          <span id="sub-edit-arrow">▲</span>
        </div>
        <div class="collapsible-content">
          <div class="form-group">
            <label for="sub-text">字幕文字內容</label>
            <input type="text" id="sub-text" placeholder="輸入要在此時間區間顯示的字幕內容...">
          </div>
          <!-- Time Range Selectors -->
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
            <div class="form-group">
              <label>開始時間 (秒)</label>
              <div style="display:flex; gap:0.35rem;">
                <input type="number" id="sub-start-sec" min="0.0" step="0.1" value="0.0" style="flex:1;">
                <button class="btn-secondary" style="padding:0.4rem 0.6rem; font-size:0.75rem;" onclick="captureCurrentVideoTime('start')" title="設定為影片當前播放秒數">📍 抓取當前</button>
              </div>
            </div>
            <div class="form-group">
              <label>結束時間 (秒)</label>
              <div style="display:flex; gap:0.35rem;">
                <input type="number" id="sub-end-sec" min="0.1" step="0.1" value="3.0" style="flex:1;">
                <button class="btn-secondary" style="padding:0.4rem 0.6rem; font-size:0.75rem;" onclick="captureCurrentVideoTime('end')" title="設定為影片當前播放秒數">📍 抓取當前</button>
              </div>
            </div>
          </div>
          <!-- Style & Position Controls -->
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
            <div class="form-group">
              <label for="sub-style">字幕樣式</label>
              <select id="sub-style">
                <option value="box" selected>🔳 半透明黑底框 (清晰推薦)</option>
                <option value="stroke">🔲 經典白字黑邊 (Classic)</option>
                <option value="yellow">🟨 高對比亮黃 (High-Vis)</option>
              </select>
            </div>
            <div class="form-group">
              <label for="sub-pos">字幕位置</label>
              <select id="sub-pos">
                <option value="bottom" selected>⬇️ 底部置中 (Bottom)</option>
                <option value="top">⬆️ 頂部置中 (Top)</option>
              </select>
            </div>
          </div>
          <div class="form-group">
            <label for="sub-size">字體大小: <span id="sub-size-val" style="color:#6366f1;">26px</span></label>
            <input type="range" id="sub-size" min="16" max="36" step="2" value="26" oninput="document.getElementById('sub-size-val').innerText=this.value+'px'">
          </div>
          <button class="btn-secondary" id="btn-burn-sub" style="background:var(--primary-gradient); color:white; border:none; padding:0.75rem; font-size:0.95rem; font-weight:700;" onclick="burnSubtitlesToCurrentVideo()">
            <span>💾 另存為帶字幕新影片 (Save Subtitled Video)</span>
          </button>
        </div>
      </div>

      <div class="metadata-grid" id="meta-grid" style="display:none;">
        <div class="meta-card">
          <div class="meta-label">解析度</div>
          <div class="meta-val" id="meta-res">768x448</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">長度 / 幀數</div>
          <div class="meta-val" id="meta-dur">2.0s (56f)</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">Seed</div>
          <div class="meta-val" id="meta-seed">42</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">Steps</div>
          <div class="meta-val" id="meta-steps">10</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">生成耗時</div>
          <div class="meta-val" id="meta-time">2.1 min</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">Peak Memory</div>
          <div class="meta-val" id="meta-mem">27.0 GB</div>
        </div>
      </div>

      <div class="action-row" id="action-row" style="display:none; display:flex; gap:0.75rem;">
        <button class="btn-secondary" style="flex:1;" onclick="openOutputFolder()">📂 開啟所在資料夾</button>
        <button class="btn-secondary" style="flex:1;" onclick="copyFilePath()">📋 複製檔案路徑</button>
        <a class="btn-secondary" id="download-link" style="flex:1;" download>⬇️ 下載 MP4</a>
      </div>
    </div>

    <!-- Bottom: History -->
    <div class="glass-card history-section">
      <div class="card-title">
        <span>🕒 最近生成紀錄 (History)</span>
      </div>
      <div class="history-grid" id="history-grid">
        <p style="color:var(--text-muted); font-size:0.85rem;">目前尚無歷史紀錄。</p>
      </div>
    </div>
  </main>

  <div id="toast" class="toast-banner"></div>

  <script>
    let currentResult = null;
    let lastToastError = null;
    let playMode = 'single';
    let serverPaths = { default: '', desktop: '', downloads: '', movies: '' };

    function showToast(msg, duration=3500) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.style.display = 'block';
      setTimeout(() => { toast.style.display = 'none'; }, duration);
    }

    function setPrompt(text) { document.getElementById('prompt').value = text; }

    function toggleDrawer(drawerId, arrowId) {
      const drawer = document.getElementById(drawerId);
      drawer.classList.toggle('open');
      document.getElementById(arrowId).innerText = drawer.classList.contains('open') ? '▲' : '▼';
    }

    function setPlayMode(mode) {
      playMode = mode;
      const player = document.getElementById('video-player');
      document.getElementById('mode-single').classList.toggle('active', mode === 'single');
      document.getElementById('mode-loop').classList.toggle('active', mode === 'loop');
      player.loop = (mode === 'loop');
    }

    function setPlaybackSpeed(speedVal) {
      const player = document.getElementById('video-player');
      player.playbackRate = parseFloat(speedVal);
    }

    function replayVideo() {
      const player = document.getElementById('video-player');
      player.currentTime = 0;
      player.play().catch(()=>{});
    }

    function captureCurrentVideoTime(target) {
      const player = document.getElementById('video-player');
      const timeVal = (player && !isNaN(player.currentTime)) ? player.currentTime.toFixed(1) : "0.0";
      if (target === 'start') {
        document.getElementById('sub-start-sec').value = timeVal;
      } else {
        document.getElementById('sub-end-sec').value = timeVal;
      }
      showToast(`已設定${target === 'start' ? '開始' : '結束'}時間為 ${timeVal} 秒`, 1500);
    }

    function alignFrameCount(rawSec) {
      const rawFrames = Math.max(5, Math.floor(rawSec * 24));
      let aligned = rawFrames;
      while (aligned % 17 !== 5) { aligned++; }
      aligned = Math.min(aligned, 362);
      const actualSec = (aligned / 24).toFixed(2);
      return { frames: aligned, sec: actualSec };
    }

    function updateDurationDisplay(val) {
      const num = parseFloat(val) || 2.0;
      const clamped = Math.max(0.5, Math.min(15.0, num));
      const info = alignFrameCount(clamped);
      document.getElementById('duration-frames-text').innerText = `換算幀數: ${info.frames} 幀 (約 ${info.sec} 秒)`;
      if (document.getElementById('sub-end-sec')) {
        document.getElementById('sub-end-sec').value = info.sec;
      }
    }

    function onDurationNumInput(val) {
      const num = parseFloat(val);
      if (!isNaN(num)) {
        document.getElementById('duration-slider').value = Math.max(0.5, Math.min(15.0, num));
        updateDurationDisplay(num);
      }
    }

    function onDurationSliderInput(val) {
      document.getElementById('duration-num').value = parseFloat(val).toFixed(1);
      updateDurationDisplay(val);
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
      if (val === serverPaths.default) document.getElementById('chip-default').classList.add('active');
      else if (val === serverPaths.desktop) document.getElementById('chip-desktop').classList.add('active');
      else if (val === serverPaths.downloads) document.getElementById('chip-downloads').classList.add('active');
      else if (val === serverPaths.movies) document.getElementById('chip-movies').classList.add('active');
    }

    function openCurrentFolder() {
      const dir = document.getElementById('output-dir').value.trim();
      fetch('/api/open-folder', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dir_path: dir})
      });
    }

    function onProfileChange() {
      const p = document.getElementById('profile').value;
      if (p === 'fast') {
        document.getElementById('width').value = 768;
        document.getElementById('height').value = 448;
        document.getElementById('duration-num').value = 2.0;
        document.getElementById('duration-slider').value = 2.0;
        document.getElementById('steps').value = 10;
      } else if (p === 'standard') {
        document.getElementById('width').value = 960;
        document.getElementById('height').value = 544;
        document.getElementById('duration-num').value = 2.5;
        document.getElementById('duration-slider').value = 2.5;
        document.getElementById('steps').value = 15;
      } else if (p === '720p_short') {
        document.getElementById('width').value = 1280;
        document.getElementById('height').value = 720;
        document.getElementById('duration-num').value = 2.0;
        document.getElementById('duration-slider').value = 2.0;
        document.getElementById('steps').value = 15;
      } else if (p === '720p') {
        document.getElementById('width').value = 1280;
        document.getElementById('height').value = 720;
        document.getElementById('duration-num').value = 3.0;
        document.getElementById('duration-slider').value = 3.0;
        document.getElementById('steps').value = 20;
      }
      updateDurationDisplay(document.getElementById('duration-num').value);
    }

    /* Queue Operations */
    async function fetchQueue() {
      try {
        const res = await fetch('/api/queue');
        if (!res.ok) return;
        const data = await res.json();
        renderQueueList(data.items || []);
        const autoChk = document.getElementById('auto-queue-chk');
        if (autoChk && typeof data.auto_enabled === 'boolean') {
          autoChk.checked = data.auto_enabled;
        }
      } catch(e) {}
    }

    function renderQueueList(items) {
      const container = document.getElementById('queue-items-container');
      const badge = document.getElementById('queue-count-badge');
      const queuedCount = items.filter(i => i.status === 'queued').length;
      badge.innerText = `${queuedCount} 筆待處理`;

      if (!items || items.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted); font-size:0.8rem; padding:0.5rem 0;">目前排程隊列為空。在上方輸入提示詞後點擊「➕ 批次加入排程」即可新增。</p>';
        return;
      }

      container.innerHTML = items.map((item, idx) => {
        const isRun = item.status === 'running';
        const isFail = item.status === 'failed';
        const isCancel = item.status === 'cancelled';
        const statusMap = {
          'queued': '⏳ 排程中',
          'running': '▶️ 生成中',
          'completed': '✅ 已完成',
          'failed': '❌ 失敗',
          'cancelled': '🛑 已取消',
          'paused': '⏸️ 已暫停'
        };
        return `
          <div class="queue-card ${item.status}">
            <div class="queue-header">
              <div style="display:flex; align-items:center; gap:0.4rem;">
                <span style="color:var(--text-muted); font-family:'JetBrains Mono',monospace;">#${idx+1}</span>
                <span class="badge-status ${item.status}">${statusMap[item.status] || item.status}</span>
                <span style="color:var(--text-muted); font-size:0.7rem;">${item.width}x${item.height} · ${item.duration_sec}s · ${item.steps}st</span>
              </div>
              <div class="queue-actions">
                ${(isFail || isCancel || item.status === 'completed') ? `<button class="btn-tiny" onclick="queueAction('${item.id}', 'retry')" title="重新排程">🔄 Retry</button>` : ''}
                ${item.status === 'queued' ? `<button class="btn-tiny" onclick="queueAction('${item.id}', 'run_now')" title="立即開始執行">⚡ 立即</button>` : ''}
                ${item.status === 'queued' ? `<button class="btn-tiny" onclick="queueAction('${item.id}', 'pause')" title="暫停">⏸️</button>` : ''}
                ${item.status === 'paused' ? `<button class="btn-tiny" onclick="queueAction('${item.id}', 'resume')" title="恢復">▶️</button>` : ''}
                <button class="btn-tiny" onclick="queueAction('${item.id}', 'delete')" title="刪除">🗑️</button>
              </div>
            </div>
            <div class="queue-prompt-text">${item.prompt}</div>
            ${item.error_message ? `<div style="font-size:0.75rem; color:#f87171; background:rgba(239,68,68,0.1); padding:0.25rem 0.5rem; border-radius:4px;">${item.error_message}</div>` : ''}
          </div>
        `;
      }).join('');
    }

    async function handleAddToQueueClick() {
      const text = document.getElementById('prompt').value.trim();
      if (!text) {
        showToast("請在上方輸入提示詞（可單筆或貼入多行批次）");
        return;
      }
      const profile = document.getElementById('profile').value;
      const width = parseInt(document.getElementById('width').value, 10) || 768;
      const height = parseInt(document.getElementById('height').value, 10) || 448;
      const duration_sec = parseFloat(document.getElementById('duration-num').value) || 2.0;
      const steps = parseInt(document.getElementById('steps').value, 10) || 10;
      const seed = parseInt(document.getElementById('seed').value, 10);
      const output_dir = document.getElementById('output-dir').value.trim();

      try {
        const res = await fetch('/api/queue/batch-add', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ prompts_text: text, profile, width, height, duration_sec, steps, seed, output_dir })
        });
        if (res.ok) {
          const data = await res.json();
          renderQueueList(data.items || []);
          showToast(`已成功加入 ${data.added_count} 筆提示詞至排程隊列！`);
        }
      } catch(e) { showToast("加入排程失敗"); }
    }

    async function queueAction(itemId, action) {
      try {
        const res = await fetch('/api/queue/action', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ item_id: itemId, action })
        });
        if (res.ok) {
          const data = await res.json();
          renderQueueList(data.items || []);
          syncJobState();
        } else {
          const err = await res.json();
          showToast(err.detail || "操作失敗");
        }
      } catch(e) {}
    }

    async function toggleAutoQueue(enabled) {
      try {
        await fetch('/api/queue/toggle-auto', { method: 'POST' });
        showToast(enabled ? "已啟用自動連續排程生成" : "已暫停自動連續生成");
      } catch(e) {}
    }

    /* Subtitle Burn-In */
    async function burnSubtitlesToCurrentVideo() {
      let videoPath = null;
      if (currentResult && (currentResult.output_path || currentResult.output_filename)) {
        videoPath = currentResult.output_path || (serverPaths.default + '/' + currentResult.output_filename);
      }
      if (!videoPath) {
        showToast("請先生成影片或從下方歷史紀錄點選一部影片進行字幕編輯");
        return;
      }
      const text = document.getElementById('sub-text').value.trim();
      if (!text) {
        showToast("請輸入字幕文字內容");
        return;
      }
      const startSec = parseFloat(document.getElementById('sub-start-sec').value) || 0.0;
      const endSec = parseFloat(document.getElementById('sub-end-sec').value) || null;
      const style = document.getElementById('sub-style').value;
      const position = document.getElementById('sub-pos').value;
      const fontSize = parseInt(document.getElementById('sub-size').value, 10) || 26;

      const btn = document.getElementById('btn-burn-sub');
      btn.disabled = true;
      btn.innerHTML = '<span>⏳ 正在快速壓制字幕... (約 1~2 秒)</span>';

      try {
        const res = await fetch('/api/subtitles/burn', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            video_path: videoPath,
            text, start_sec: startSec, end_sec: endSec,
            style, position, font_size: fontSize
          })
        });
        if (!res.ok) {
          const err = await res.json();
          showToast("字幕壓制失敗: " + (err.detail || "未知錯誤"));
          return;
        }
        const data = await res.json();
        showToast("✨ 帶字幕新影片另存成功！");
        // Switch player to newly subtitled video
        const newResult = Object.assign({}, currentResult, {
          output_path: data.output_path,
          output_filename: data.output_filename
        });
        displayResult(newResult);
        fetchHistory();
      } catch (e) {
        showToast("壓制失敗，請確認檔案路徑");
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>💾 另存為帶字幕新影片 (Save Subtitled Video)</span>';
      }
    }

    async function cancelCurrentGeneration() {
      try {
        const btn = document.getElementById('btn-cancel');
        btn.innerText = '中止中... (Cancelling)';
        btn.disabled = true;
        await fetch('/api/generate/cancel', { method: 'POST' });
        showToast("已發送中止訊號，正在釋放 Metal 記憶體...");
      } catch(e) {}
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        const mem = data.memory;
        const pDot = document.getElementById('pressure-dot');
        const pText = document.getElementById('pressure-text');
        const swapPill = document.getElementById('swap-pill');

        pText.innerText = `RAM: ${mem.pressure} (Free ${mem.free_pct}%)`;
        swapPill.innerText = `Swap: ${mem.swap_used_gib} GB`;
        pDot.className = 'indicator-dot ' + (mem.pressure === 'Warning' ? 'warning' : (mem.pressure === 'Critical' ? 'danger' : ''));

        if (!serverPaths.default) {
          serverPaths.default = data.default_output_dir;
          serverPaths.desktop = data.desktop_dir;
          serverPaths.downloads = data.downloads_dir;
          serverPaths.movies = data.movies_dir;

          const saved = localStorage.getItem('minimax_output_dir');
          if (saved) {
            document.getElementById('output-dir').value = saved;
            saveOutputDir(saved);
          } else {
            document.getElementById('output-dir').value = data.default_output_dir;
          }
        }
      } catch(e) {}
    }

    async function syncJobState() {
      try {
        const res = await fetch('/api/job');
        if (!res.ok) return;
        const job = await res.json();

        if (job.is_running) {
          showRunning(job);
        } else {
          showIdle();
          if (job.result) {
            if (!currentResult || currentResult.output_filename !== job.result.output_filename) {
              displayResult(job.result);
              fetchHistory();
            }
          } else if (job.error) {
            if (lastToastError !== job.error) {
              lastToastError = job.error;
              showToast("任務中斷: " + job.error);
            }
          }
        }
      } catch (e) {}
    }

    async function fetchHistory() {
      try {
        const res = await fetch('/api/history');
        if (!res.ok) return [];
        const items = await res.json();
        const grid = document.getElementById('history-grid');
        if (!items || items.length === 0) {
          grid.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem;">目前尚無歷史紀錄。</p>';
          return [];
        }
        grid.innerHTML = items.map(item => {
          const videoSrc = item.output_path ? `/api/video-stream?path=${encodeURIComponent(item.output_path)}` : (item.output_filename ? `/outputs/${item.output_filename}` : '');
          return `
          <div class="history-item" onclick='loadHistoryItem(${JSON.stringify(item)})'>
            <div class="history-video-thumb">
              ${videoSrc ? `<video src="${videoSrc}" preload="metadata" muted></video>` : '❌ Failed'}
            </div>
            <div class="history-prompt">${item.prompt}</div>
            <div class="history-footer">
              <span>${item.width}x${item.height} (${item.duration_sec}s)</span>
              <span>${item.execution_time_sec ? (item.execution_time_sec/60).toFixed(1)+'m' : ''}</span>
            </div>
          </div>
        `}).join('');
        return items;
      } catch (e) { return []; }
    }

    function loadHistoryItem(item) {
      if (!item || (!item.output_path && !item.output_filename)) return;
      displayResult(item);
    }

    function displayResult(res) {
      currentResult = res;
      const player = document.getElementById('video-player');
      const empty = document.getElementById('empty-state');
      const toolbar = document.getElementById('player-toolbar');

      empty.style.display = 'none';
      player.style.display = 'block';
      toolbar.style.display = 'flex';

      player.loop = (playMode === 'loop');
      player.playbackRate = parseFloat(document.getElementById('play-speed').value) || 1.0;

      const videoUrl = res.output_path ? `/api/video-stream?path=${encodeURIComponent(res.output_path)}` : `/outputs/${res.output_filename}`;
      if (player.src !== window.location.origin + videoUrl && player.getAttribute('src') !== videoUrl) {
        player.src = videoUrl;
        player.play().catch(()=>{});
      }

      document.getElementById('meta-res').innerText = `${res.width}x${res.height}`;
      document.getElementById('meta-dur').innerText = `${res.duration_sec}s (${res.frames}f)`;
      document.getElementById('meta-seed').innerText = res.seed;
      document.getElementById('meta-steps').innerText = res.steps;
      document.getElementById('meta-time').innerText = (res.execution_time_sec / 60).toFixed(1) + ' min';
      document.getElementById('meta-mem').innerText = (res.peak_memory_gib || '27.0') + ' GB';

      // Set default subtitle end time to match duration
      if (res.duration_sec) {
        document.getElementById('sub-end-sec').value = res.duration_sec;
      }

      document.getElementById('meta-grid').style.display = 'grid';
      document.getElementById('action-row').style.display = 'flex';
      document.getElementById('download-link').href = videoUrl;
    }

    function showRunning(job) {
      document.getElementById('btn-gen').style.display = 'none';
      document.getElementById('btn-add-single-queue').style.display = 'none';
      const cancelBtn = document.getElementById('btn-cancel');
      cancelBtn.style.display = 'flex';
      cancelBtn.disabled = false;
      cancelBtn.innerHTML = '<span>🛑 中止生成 (Cancel)</span>';

      document.getElementById('progress-box').style.display = 'flex';
      document.getElementById('stage-text').innerText = job.stage || 'Processing...';
      const pct = Math.min(100, Math.max(5, (job.progress * 100)));
      document.getElementById('progress-bar').style.width = pct + '%';
      const pctEl = document.getElementById('stage-pct');
      if (pctEl) pctEl.innerText = Math.floor(pct) + '%';
      if (job.elapsed_sec) {
        const m = Math.floor(job.elapsed_sec / 60);
        const s = Math.floor(job.elapsed_sec % 60);
        document.getElementById('time-text').innerText = m > 0 ? `${m}m ${s}s` : `${s}s`;
      } else {
        document.getElementById('time-text').innerText = '0s';
      }
    }

    function showIdle() {
      document.getElementById('btn-gen').style.display = 'flex';
      document.getElementById('btn-add-single-queue').style.display = 'flex';
      document.getElementById('btn-cancel').style.display = 'none';
      document.getElementById('progress-box').style.display = 'none';
    }

    async function handleGenerateClick() {
      const fullText = document.getElementById('prompt').value.trim();
      if (!fullText) { showToast("請輸入提示詞 (Prompt)"); return; }

      const lines = fullText.split('\\n').map(l => l.trim()).filter(l => l.length > 0);
      const profile = document.getElementById('profile').value;
      const width = parseInt(document.getElementById('width').value, 10) || 768;
      const height = parseInt(document.getElementById('height').value, 10) || 448;
      const duration_sec = parseFloat(document.getElementById('duration-num').value) || 2.0;
      const steps = parseInt(document.getElementById('steps').value, 10) || 10;
      const seed = parseInt(document.getElementById('seed').value, 10);
      const output_dir = document.getElementById('output-dir').value.trim();

      // If user provided multiple lines in the prompt box, immediately run the first one and queue the rest!
      if (lines.length > 1) {
        const restLines = lines.slice(1).join('\\n');
        try {
          await fetch('/api/queue/batch-add', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompts_text: restLines, profile, width, height, duration_sec, steps, seed, output_dir })
          });
          fetchQueue();
        } catch (e) {}
      }

      const promptToRun = lines[0];
      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ prompt: promptToRun, profile, width, height, duration_sec, steps, seed, output_dir })
        });
        if (!res.ok) {
          const err = await res.json();
          showToast("啟動失敗: " + (err.detail || "未知錯誤"));
          return;
        }
        lastToastError = null;
        showRunning({stage: 'Starting generation...', progress: 0.05, elapsed_sec: 0});
        syncJobState();
      } catch (e) {
        showToast("⚠️ 無法連接到本地伺服器，請確認服務已啟動。");
        showIdle();
      }
    }

    async function openOutputFolder() {
      if (currentResult && currentResult.output_path) {
        await fetch('/api/open-folder', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({file_path: currentResult.output_path})
        });
      } else {
        const dir = document.getElementById('output-dir').value.trim();
        await fetch('/api/open-folder', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({dir_path: dir})
        });
      }
    }

    function copyFilePath() {
      if (currentResult && currentResult.output_path) {
        navigator.clipboard.writeText(currentResult.output_path);
        showToast("已複製檔案路徑至剪貼簿！", 2500);
      }
    }

    // Initialize on page load
    async function initApp() {
      updateDurationDisplay(2.0);
      await fetchStatus();
      await fetchQueue();
      await syncJobState();
      const historyList = await fetchHistory();
      if (!currentResult && historyList && historyList.length > 0 && historyList[0].output_filename) {
        displayResult(historyList[0]);
      }
      setInterval(syncJobState, 1500);
      setInterval(fetchQueue, 2500);
      setInterval(fetchStatus, 4000);
    }

    window.addEventListener('DOMContentLoaded', initApp);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return INDEX_HTML


def start_server(port: int | None = None, open_browser: bool = True):
    actual_port = port or 7860
    url = f"http://127.0.0.1:{actual_port}"

    print("\n" + "=" * 60)
    print("🎬 MiniMax-H3 Local Web Studio (with Prompt Queue & Subtitle Editor)")
    print(f"URL: {url}")
    print("=" * 60 + "\n", flush=True)

    if open_browser:
        def _open():
            time.sleep(1.0)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=actual_port, log_level="info")


if __name__ == "__main__":
    start_server()
