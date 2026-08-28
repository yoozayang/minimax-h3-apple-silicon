#!/usr/bin/env python3
"""Unified Model Residency & Dynamic Model Swapping Manager for Apple Silicon."""

from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Any

import mlx.core as mx

MODEL_LOCK = threading.RLock()

# Active engine state: "NONE", "IMAGE", "VIDEO"
_ACTIVE_ENGINE: str = "NONE"
_LOADED_IMAGE_MODEL: Any = None
_UNLOAD_CALLBACKS: dict[str, list[Callable[[], None]]] = {
    "IMAGE": [],
    "VIDEO": [],
}


def get_active_engine() -> str:
    """Return the currently resident engine name."""
    global _ACTIVE_ENGINE
    with MODEL_LOCK:
        return _ACTIVE_ENGINE


def register_unload_callback(engine: str, callback: Callable[[], None]) -> None:
    """Register a hook to be called when an engine is being unloaded."""
    with MODEL_LOCK:
        if engine in _UNLOAD_CALLBACKS:
            _UNLOAD_CALLBACKS[engine].append(callback)


def clear_metal_memory() -> None:
    """Aggressively collect Python garbage and clear MLX/Metal memory cache."""
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def switch_to_engine(target: str) -> None:
    """Switch active engine to target ('IMAGE' or 'VIDEO'), unloading previous models."""
    global _ACTIVE_ENGINE, _LOADED_IMAGE_MODEL
    target = target.upper()
    with MODEL_LOCK:
        if _ACTIVE_ENGINE == target:
            return

        print(f"[ModelManager] Switching active engine from '{_ACTIVE_ENGINE}' to '{target}'...")

        # 1. Unload current engine
        if _ACTIVE_ENGINE == "IMAGE":
            for cb in _UNLOAD_CALLBACKS.get("IMAGE", []):
                try:
                    cb()
                except Exception as e:
                    print(f"[ModelManager] Image unload hook error: {e}", file=sys.stderr)
            _LOADED_IMAGE_MODEL = None
            clear_metal_memory()
            print("[ModelManager] Image Engine unloaded. Metal memory cleared.")

        elif _ACTIVE_ENGINE == "VIDEO":
            for cb in _UNLOAD_CALLBACKS.get("VIDEO", []):
                try:
                    cb()
                except Exception as e:
                    print(f"[ModelManager] Video unload hook error: {e}", file=sys.stderr)
            clear_metal_memory()
            print("[ModelManager] Video Engine components unloaded. Metal memory cleared.")

        # 2. Update active engine
        _ACTIVE_ENGINE = target
        print(f"[ModelManager] Active engine is now '{_ACTIVE_ENGINE}'.")


def release_all() -> None:
    """Unload all resident models and reset to NONE."""
    global _ACTIVE_ENGINE, _LOADED_IMAGE_MODEL
    with MODEL_LOCK:
        for eng in ("IMAGE", "VIDEO"):
            for cb in _UNLOAD_CALLBACKS.get(eng, []):
                try:
                    cb()
                except Exception:
                    pass
        _LOADED_IMAGE_MODEL = None
        clear_metal_memory()
        _ACTIVE_ENGINE = "NONE"
        print("[ModelManager] All models released and memory cleared.")
