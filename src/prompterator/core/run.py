"""Run directory management for output isolation."""

from datetime import datetime
from pathlib import Path


def _next_run_seq(base: Path) -> int:
    """Find the next run sequence number under *base*."""
    import re

    pattern = re.compile(r"^run(\d{3})-")
    max_seq = 0
    if base.exists():
        for entry in base.iterdir():
            m = pattern.match(entry.name)
            if m:
                max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1


def create_run_dir(base: Path) -> Path:
    """Create a sequenced, timestamped run directory under *base*.

    Format: ``<base>/run001-YYYYMMDD-HHMMSS/``

    If debug logging is enabled, debug files are written into this directory.
    """
    base.mkdir(parents=True, exist_ok=True)
    seq = _next_run_seq(base)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"run{seq:03d}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Point debug logs into this run directory if enabled
    from prompterator.runners.llm import _debug_enabled, set_debug_log_dir

    if _debug_enabled:
        set_debug_log_dir(run_dir)

    return run_dir
