#!/usr/bin/env python3
"""Lightweight local Web UI for MiniMax-H3 on Apple Silicon."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path

# Dynamically resolve project directory
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import engine
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="MiniMax-H3 Local Web UI")

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
}
JOB_LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    profile: str = "fast"
    width: int = 768
    height: int = 448
    duration_sec: float = 2.0
    seed: int = -1
    steps: int = 10
    output_dir: str = ""
    subtitle_text: str = ""
    subtitle_position: str = "bottom"
    subtitle_style: str = "box"
    subtitle_font_size: int = 24


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(start_port: int = 7860, max_port: int = 7890) -> int:
    for port in range(start_port, max_port + 1):
        if not is_port_in_use(port):
            return port
    return start_port


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
        }
    return {
        "memory": mem,
        "job": job_info,
        "profiles": engine.PROFILES,
        "default_output_dir": str(OUTPUTS_DIR),
        "desktop_dir": str(Path("~/Desktop").expanduser().resolve()),
        "downloads_dir": str(Path("~/Downloads").expanduser().resolve()),
        "movies_dir": str(Path("~/Movies").expanduser().resolve()),
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
        }


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


@app.post("/api/generate")
async def start_generation(req: GenerateRequest):
    global CURRENT_JOB
    with JOB_LOCK:
        if CURRENT_JOB["is_running"]:
            raise HTTPException(status_code=409, detail="已有生成任務正在執行中，請稍候。")
        CURRENT_JOB["is_running"] = True
        CURRENT_JOB["stage"] = "Starting generation..."
        CURRENT_JOB["progress"] = 0.01
        CURRENT_JOB["result"] = None
        CURRENT_JOB["error"] = None
        CURRENT_JOB["started_at"] = time.time()
        CURRENT_JOB["prompt"] = req.prompt.strip()
        CURRENT_JOB["width"] = req.width
        CURRENT_JOB["height"] = req.height
        CURRENT_JOB["duration_sec"] = req.duration_sec
        CURRENT_JOB["steps"] = req.steps
        CURRENT_JOB["seed"] = req.seed
        CURRENT_JOB["output_dir"] = req.output_dir

    def run_task():
        def on_stage(stage_text: str, progress_val: float):
            with JOB_LOCK:
                CURRENT_JOB["stage"] = stage_text
                CURRENT_JOB["progress"] = progress_val

        try:
            w = req.width
            h = req.height
            dur = max(0.5, min(15.08, req.duration_sec))
            st = max(4, min(60, req.steps))
            seed_val = req.seed if req.seed >= 0 else None
            custom_out = req.output_dir.strip() if req.output_dir and req.output_dir.strip() else str(OUTPUTS_DIR)

            res = engine.generate_video(
                prompt=req.prompt,
                width=w,
                height=h,
                duration_sec=dur,
                seed=seed_val,
                steps=st,
                output_dir=custom_out,
                subtitle_text=req.subtitle_text,
                subtitle_position=req.subtitle_position,
                subtitle_style=req.subtitle_style,
                subtitle_font_size=req.subtitle_font_size,
                on_stage=on_stage,
            )
            with JOB_LOCK:
                CURRENT_JOB["is_running"] = False
                CURRENT_JOB["progress"] = 1.0
                if res.success:
                    CURRENT_JOB["stage"] = "Completed successfully!"
                    CURRENT_JOB["result"] = engine.asdict(res)
                else:
                    CURRENT_JOB["stage"] = "Generation failed."
                    CURRENT_JOB["error"] = res.error_message
        except Exception as e:
            with JOB_LOCK:
                CURRENT_JOB["is_running"] = False
                CURRENT_JOB["stage"] = "Generation failed."
                CURRENT_JOB["error"] = str(e)
                CURRENT_JOB["progress"] = 0.0

    thread = threading.Thread(target=run_task, daemon=True)
    thread.start()
    return {"status": "started"}


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


# Static mount for default output dir
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# Modern Single-Page App HTML
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-TW" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MiniMax-H3 Local Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f17;
      --card-bg: rgba(20, 27, 41, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary-accent: #6366f1;
      --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --surface-hover: rgba(255, 255, 255, 0.04);
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
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand-badge {
      background: var(--primary-gradient);
      width: 38px;
      height: 38px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.2rem;
      color: white;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    .brand-text h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand-text p {
      font-size: 0.75rem;
      color: var(--text-muted);
    }
    .system-status {
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 0.8rem;
    }
    .status-pill {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-family: 'JetBrains Mono', monospace;
    }
    .indicator-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }
    .indicator-dot.warning { background: var(--warning); box-shadow: 0 0 8px var(--warning); }
    .indicator-dot.danger { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

    main {
      flex: 1;
      max-width: 1300px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      width: 100%;
      display: grid;
      grid-template-columns: 1fr 1.1fr;
      gap: 2rem;
    }
    @media (max-width: 992px) {
      main { grid-template-columns: 1fr; }
    }

    .glass-card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .card-title {
      font-size: 1.1rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 0.75rem;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    textarea, select, input {
      background: rgba(11, 15, 23, 0.7);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 0.75rem 1rem;
      border-radius: 10px;
      font-family: inherit;
      font-size: 0.95rem;
      transition: all 0.2s ease;
      outline: none;
    }
    textarea:focus, select:focus, input:focus {
      border-color: var(--primary-accent);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
    }
    textarea {
      min-height: 100px;
      resize: vertical;
      line-height: 1.5;
    }

    .chips-container {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      font-size: 0.75rem;
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      cursor: pointer;
      color: var(--text-muted);
      transition: 0.15s ease;
    }
    .chip:hover {
      background: rgba(99, 102, 241, 0.2);
      color: var(--text-main);
      border-color: var(--primary-accent);
    }
    .chip.active {
      background: rgba(99, 102, 241, 0.3);
      color: #a5b4fc;
      border-color: #818cf8;
      font-weight: 600;
    }

    .duration-control-box, .path-control-box, .subtitle-control-box {
      background: rgba(11, 15, 23, 0.5);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .duration-inputs {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .duration-num-input {
      width: 100px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 1.05rem;
      text-align: center;
      color: #818cf8;
    }
    .duration-slider {
      flex: 1;
      height: 6px;
      accent-color: #6366f1;
      cursor: pointer;
    }
    .duration-calc-badge {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      display: flex;
      justify-content: space-between;
    }

    .path-row {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    .path-input {
      flex: 1;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      padding: 0.6rem 0.8rem;
    }

    .collapsible {
      border: 1px solid var(--card-border);
      border-radius: 10px;
      overflow: hidden;
      background: rgba(11, 15, 23, 0.4);
    }
    .collapsible-header {
      padding: 0.75rem 1rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .collapsible-content {
      padding: 1rem;
      display: none;
      flex-direction: column;
      gap: 1rem;
      border-top: 1px solid var(--card-border);
    }
    .collapsible.open .collapsible-content {
      display: flex;
    }

    .btn-generate {
      background: var(--primary-gradient);
      border: none;
      color: white;
      font-size: 1.05rem;
      font-weight: 700;
      padding: 1rem;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.25s ease;
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
    }
    .btn-generate:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6);
    }
    .btn-generate:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .progress-box {
      background: rgba(11, 15, 23, 0.9);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1rem;
      display: none;
      flex-direction: column;
      gap: 0.75rem;
    }
    .progress-bar-bg {
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%;
      background: var(--primary-gradient);
      width: 0%;
      transition: width 0.3s ease;
    }
    .stage-text {
      font-size: 0.85rem;
      color: #93c5fd;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Player and Result */
    .player-container {
      border-radius: 12px;
      overflow: hidden;
      background: #000;
      border: 1px solid var(--card-border);
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .empty-state {
      color: var(--text-muted);
      font-size: 0.9rem;
      text-align: center;
      padding: 2rem;
    }

    /* Player Controls Bar */
    .player-control-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(11, 15, 23, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.5rem 0.85rem;
      font-size: 0.8rem;
    }
    .play-mode-chips {
      display: flex;
      gap: 0.35rem;
      align-items: center;
    }
    .play-chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      padding: 0.25rem 0.55rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.75rem;
      color: var(--text-muted);
      transition: 0.15s ease;
    }
    .play-chip:hover {
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.1);
    }
    .play-chip.active {
      background: rgba(99, 102, 241, 0.35);
      color: #c7d2fe;
      border-color: #818cf8;
      font-weight: 600;
    }

    .metadata-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
    }
    .meta-card {
      background: rgba(11, 15, 23, 0.6);
      border: 1px solid var(--card-border);
      padding: 0.6rem;
      border-radius: 8px;
      text-align: center;
    }
    .meta-label {
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .meta-val {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text-main);
      font-family: 'JetBrains Mono', monospace;
      margin-top: 0.2rem;
    }

    .action-row {
      display: flex;
      gap: 0.75rem;
    }
    .btn-secondary {
      flex: 1;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 0.6rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      text-decoration: none;
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.2);
    }

    /* History section */
    .history-section {
      grid-column: 1 / -1;
      margin-top: 1rem;
    }
    .history-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1rem;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 0.5rem;
    }
    .history-item {
      background: rgba(11, 15, 23, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .history-item:hover {
      border-color: var(--primary-accent);
      transform: translateY(-2px);
    }
    .history-video-thumb {
      width: 100%;
      height: 140px;
      background: #000;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .history-prompt {
      font-size: 0.8rem;
      color: var(--text-main);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      line-height: 1.3;
    }
    .history-footer {
      display: flex;
      justify-content: space-between;
      font-size: 0.7rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }

    .toast-banner {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: rgba(239, 68, 68, 0.95);
      color: white;
      padding: 0.75rem 1.25rem;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      display: none;
      z-index: 100;
      font-size: 0.9rem;
      font-weight: 600;
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
        <span id="pressure-text">Memory: Checking...</span>
      </div>
      <div class="status-pill" id="swap-pill">
        <span>Swap: 0.0 GB</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Left: Generation Controls -->
    <div class="glass-card">
      <div class="card-title">
        <span>✨ 提示詞與生成設定</span>
      </div>

      <div class="form-group">
        <label for="prompt">Prompt (提示詞)</label>
        <textarea id="prompt" placeholder="A corgi running through a vibrant grassy field, golden hour lighting, cinematic camera movement, high detail..."></textarea>
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
            <span>架構上限: 15.0 秒 (362 幀)</span>
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

      <!-- Subtitles Drawer -->
      <div class="collapsible" id="subtitle-drawer">
        <div class="collapsible-header" onclick="toggleDrawer('subtitle-drawer', 'sub-arrow')">
          <span>💬 影片字幕設定 (Subtitle & Caption)</span>
          <span id="sub-arrow">▼</span>
        </div>
        <div class="collapsible-content">
          <div class="form-group">
            <label for="sub-text">字幕文字內容 (留空代表不加字幕)</label>
            <input type="text" id="sub-text" placeholder="例如：奔跑在陽光草地上的柯基犬...">
            <div class="chips-container" style="margin-top:0.25rem;">
              <span class="chip" onclick="copyPromptToSubtitle()">📋 複製提示詞為字幕</span>
              <span class="chip" onclick="document.getElementById('sub-text').value=''">❌ 清空字幕</span>
            </div>
          </div>
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
            <label for="sub-size">字體大小: <span id="sub-size-val" style="color:#6366f1;">24px</span></label>
            <input type="range" id="sub-size" min="16" max="36" step="2" value="24" oninput="document.getElementById('sub-size-val').innerText=this.value+'px'">
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
              <label for="width">寬度 (Width, 16倍數)</label>
              <input type="number" id="width" value="768" step="16" min="128" max="1344">
            </div>
            <div class="form-group">
              <label for="height">高度 (Height, 16倍數)</label>
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
          <p style="font-size:0.75rem; color:var(--text-muted);">
            💡 註：MiniMax-H3 為 CFG-distilled 擴散架構，已將引導蒸餾至權重中，無需輸入 Negative Prompt。
          </p>
        </div>
      </div>

      <button class="btn-generate" id="btn-gen" onclick="startGenerate()">
        <span id="btn-icon">🚀</span>
        <span id="btn-text">開始生成 (Generate)</span>
      </button>

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
    </div>

    <!-- Right: Result Showcase & Live Output -->
    <div class="glass-card">
      <div class="card-title">
        <span>🎬 生成結果與播放器</span>
      </div>

      <div class="player-container" id="player-wrap">
        <div class="empty-state" id="empty-state">
          點擊「開始生成」後，影片將即時在此播放
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

      <div class="metadata-grid" id="meta-grid" style="display:none;">
        <div class="meta-card">
          <div class="meta-label">解析度</div>
          <div class="meta-val" id="meta-res">768x448</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">長度 / 幀數</div>
          <div class="meta-val" id="meta-dur">2.0s (16f)</div>
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
          <div class="meta-val" id="meta-mem">21.8 GB</div>
        </div>
      </div>

      <div class="action-row" id="action-row" style="display:none;">
        <button class="btn-secondary" onclick="openOutputFolder()">📂 開啟所在資料夾</button>
        <button class="btn-secondary" onclick="copyFilePath()">📋 複製檔案路徑</button>
        <a class="btn-secondary" id="download-link" download>⬇️ 下載 MP4</a>
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
    let playMode = 'single'; // default: single play
    let serverPaths = {
      default: '',
      desktop: '',
      downloads: '',
      movies: ''
    };

    function showToast(msg, duration=3500) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.style.display = 'block';
      setTimeout(() => { toast.style.display = 'none'; }, duration);
    }

    function setPrompt(text) {
      document.getElementById('prompt').value = text;
    }

    function copyPromptToSubtitle() {
      document.getElementById('sub-text').value = document.getElementById('prompt').value.trim();
    }

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

    function alignFrameCount(rawSec) {
      const rawFrames = Math.max(5, Math.floor(rawSec * 24));
      let aligned = rawFrames;
      while (aligned % 17 !== 5) {
        aligned++;
      }
      aligned = Math.min(aligned, 362);
      const actualSec = (aligned / 24).toFixed(2);
      return { frames: aligned, sec: actualSec };
    }

    function updateDurationDisplay(val) {
      const num = parseFloat(val) || 2.0;
      const clamped = Math.max(0.5, Math.min(15.0, num));
      const info = alignFrameCount(clamped);
      document.getElementById('duration-frames-text').innerText = `換算幀數: ${info.frames} 幀 (約 ${info.sec} 秒)`;
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
          if (job.prompt && !document.getElementById('prompt').value) {
            document.getElementById('prompt').value = job.prompt;
          }
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
              showToast("生成失敗: " + job.error);
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
      } catch (e) {
        return [];
      }
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

      // Set default single playback mode
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

      document.getElementById('meta-grid').style.display = 'grid';
      document.getElementById('action-row').style.display = 'flex';
      document.getElementById('download-link').href = videoUrl;
    }

    function showRunning(job) {
      document.getElementById('btn-gen').disabled = true;
      document.getElementById('btn-text').innerText = '生成中... (Generating)';
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
      document.getElementById('btn-gen').disabled = false;
      document.getElementById('btn-text').innerText = '開始生成 (Generate)';
      document.getElementById('progress-box').style.display = 'none';
    }

    async function startGenerate() {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) {
        showToast("請輸入提示詞 (Prompt)");
        return;
      }
      const profile = document.getElementById('profile').value;
      const width = parseInt(document.getElementById('width').value, 10) || 768;
      const height = parseInt(document.getElementById('height').value, 10) || 448;
      const duration_sec = parseFloat(document.getElementById('duration-num').value) || 2.0;
      const steps = parseInt(document.getElementById('steps').value, 10) || 10;
      const seed = parseInt(document.getElementById('seed').value, 10);
      const output_dir = document.getElementById('output-dir').value.trim();

      // Subtitle parameters
      const subtitle_text = document.getElementById('sub-text').value.trim();
      const subtitle_style = document.getElementById('sub-style').value;
      const subtitle_position = document.getElementById('sub-pos').value;
      const subtitle_font_size = parseInt(document.getElementById('sub-size').value, 10) || 24;

      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            prompt, profile, width, height, duration_sec, steps, seed, output_dir,
            subtitle_text, subtitle_style, subtitle_position, subtitle_font_size
          })
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
        showToast("⚠️ 無法連接到本地伺服器，請確認『色色』已啟動。");
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
      await syncJobState();
      const historyList = await fetchHistory();
      if (!currentResult && historyList && historyList.length > 0 && historyList[0].output_filename) {
        displayResult(historyList[0]);
      }
      setInterval(syncJobState, 1500);
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
    print("🎬 MiniMax-H3 Local Web Studio")
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
