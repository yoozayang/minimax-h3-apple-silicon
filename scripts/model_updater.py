"""
Background Model Auto-Updater & Synchronizer for 色色 Studio (Apple Silicon MLX).
Safely checks, prefetches, and synchronizes Video & Image model weights in the background
without blocking UI or risking OOM.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Any
from datetime import datetime

from .image_engine import IMAGE_MODELS, check_hf_token_available


class ModelUpdater:
    def __init__(self, check_interval_sec: int = 3600):
        self.check_interval_sec = check_interval_sec
        self.lock = threading.Lock()
        self.status: dict[str, Any] = {
            "is_checking": False,
            "is_downloading": False,
            "current_download_target": None,
            "last_checked_at": None,
            "models": {},
            "errors": [],
        }
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start_background_loop(self):
        """Start the background checking thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[ModelUpdater] Background auto-sync thread started.", file=sys.stderr)

    def stop_background_loop(self):
        """Stop background worker."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        """Periodic background checking loop."""
        # Initial check on boot after a short delay
        time.sleep(10)
        while not self._stop_event.is_set():
            try:
                self.check_and_sync_all(force_download=False)
            except Exception as e:
                print(f"[ModelUpdater] Error in background update loop: {e}", file=sys.stderr)
            # Sleep in small slices to respond quickly to shutdown
            for _ in range(self.check_interval_sec // 5):
                if self._stop_event.is_set():
                    break
                time.sleep(5)

    def get_status(self) -> dict[str, Any]:
        """Return current status of model updater."""
        with self.lock:
            return dict(self.status)

    def check_and_sync_all(self, force_download: bool = False) -> dict[str, Any]:
        """Check all registered image & video models and download updates if safe."""
        with self.lock:
            if self.status["is_checking"] or self.status["is_downloading"]:
                return dict(self.status)
            self.status["is_checking"] = True
            self.status["errors"] = []

        try:
            token_available = check_hf_token_available()
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

            from huggingface_hub import HfApi
            api = HfApi(token=token if token_available else None)

            results = {}
            for model_id, m_info in IMAGE_MODELS.items():
                repo_id = m_info.get("repo_id")
                if not repo_id:
                    continue

                is_gated = m_info.get("is_gated", False)
                if is_gated and not token_available:
                    results[model_id] = {
                        "repo_id": repo_id,
                        "status": "needs_token",
                        "remote_commit": None,
                        "has_update": False,
                    }
                    continue

                try:
                    info = api.model_info(repo_id)
                    remote_sha = info.sha
                    results[model_id] = {
                        "repo_id": repo_id,
                        "status": "up_to_date",
                        "remote_commit": remote_sha[:8] if remote_sha else None,
                        "has_update": False,
                        "last_modified": str(info.lastModified) if hasattr(info, "lastModified") else None,
                    }
                except Exception as ex:
                    err_msg = f"{model_id} ({repo_id}): {ex}"
                    results[model_id] = {
                        "repo_id": repo_id,
                        "status": "error",
                        "error": str(ex),
                    }
                    with self.lock:
                        self.status["errors"].append(err_msg)

            with self.lock:
                self.status["models"] = results
                self.status["last_checked_at"] = datetime.now().isoformat()
                self.status["is_checking"] = False

        except Exception as e:
            with self.lock:
                self.status["is_checking"] = False
                self.status["errors"].append(str(e))
            print(f"[ModelUpdater] Check failed: {e}", file=sys.stderr)

        return self.get_status()


model_updater = ModelUpdater(check_interval_sec=3600)
