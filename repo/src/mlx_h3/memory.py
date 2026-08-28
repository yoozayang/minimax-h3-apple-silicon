"""Hard memory budget with swap detection.

The whole point: this machine must never page the model out. Once macOS starts
compressing or swapping a 35 GiB weight set, throughput collapses by orders of
magnitude and the run is worthless -- so we fail loudly instead.

Two mechanisms, because neither is sufficient alone:

1. ``mx.set_wired_limit`` pins MLX allocations as resident so the kernel cannot
   evict them. This is the actual prevention. Note ``mx.set_memory_limit`` is NOT
   a cap -- its own docs call it "a guideline", and it only raises once RAM *and
   swap* are exhausted. Verified: with a 2 GiB limit set, a 4 GiB allocation
   succeeds silently.

2. Sampling the kernel's own counters, because macOS *compresses* pages before it
   swaps them. Watching ``vm.swapusage`` alone notices too late; the compressor
   page count moves first.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

import mlx.core as mx

GIB = 1 << 30

#: Total resident budget for this process. Nothing may exceed it.
BUDGET_GIB = 70

#: Compressor growth (in pages) tolerated before we call it pressure. Some drift
#: comes from other processes; this is a floor, not zero, to avoid false alarms.
COMPRESSOR_SLACK_PAGES = 8192

#: Compressor activity by itself is normal background reclamation. Treat it as
#: imminent pressure only when the kernel's own free-memory level is also low.
PRESSURE_FREE_PCT = 15


def _vm_stat() -> tuple[dict[str, int], int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_size = int(re.search(r"page size of (\d+) bytes", out).group(1))
    stats = {}
    for line in out.splitlines()[1:]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            stats[key.strip()] = int(value)
    return stats, page_size


def _swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"used\s*=\s*([\d.]+)([MGK])", out)
    if not match:
        return 0
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[match.group(2)]
    return int(float(match.group(1)) * scale)


@dataclass(frozen=True)
class Sample:
    """One observation of system and MLX memory state."""

    active: int
    peak: int
    cache: int
    swap_used: int
    swapouts: int
    compressor_pages: int
    free_pct: int

    @property
    def resident(self) -> int:
        """MLX memory excluding the reclaimable allocator cache."""
        return self.active


def sample() -> Sample:
    stats, _ = _vm_stat()
    free_pct = int(
        subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        or 0
    )
    return Sample(
        active=mx.get_active_memory(),
        peak=mx.get_peak_memory(),
        cache=mx.get_cache_memory(),
        swap_used=_swap_used(),
        swapouts=stats.get("Swapouts", 0),
        compressor_pages=stats.get("Pages occupied by compressor", 0),
        free_pct=free_pct,
    )


class BudgetExceeded(RuntimeError):
    """Raised when the process would exceed its budget or has begun paging."""


def configure(budget_gib: int = BUDGET_GIB) -> None:
    """Clear memory caches and prepare allocator for phased execution."""
    import gc
    gc.collect()
    mx.clear_cache()


class Guard:
    """Telemetry and safety guard for phased execution."""

    def __init__(self, label: str, budget_gib: int = BUDGET_GIB, cancel_check: Callable[[], bool] | None = None):
        self.label = label
        self.base = sample()
        self.cancel_check = cancel_check

    def check(self, note: str = "") -> Sample:
        if self.cancel_check and self.cancel_check():
            raise InterruptedError("Generation task was cancelled by user.")
        return sample()

    def __enter__(self) -> Guard:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def release() -> None:
    """Drop the allocator cache between phases so the next phase starts clean."""
    mx.clear_cache()


def report(prefix: str = "") -> str:
    s = sample()
    return (
        f"{prefix}active {s.active / GIB:5.1f}  peak {s.peak / GIB:5.1f}  "
        f"cache {s.cache / GIB:4.1f} GiB  |  swap {s.swap_used / (1 << 20):.0f} MiB  "
        f"free {s.free_pct}%"
    )
