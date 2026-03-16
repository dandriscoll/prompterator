"""Live progress counter for interactive CLI output."""

from __future__ import annotations

import sys
import time


class Progress:
    """Tracks LLM call progress and prints a live status line to stderr.

    Usage::

        progress = Progress(total=36, label="Testing")
        # ... in a loop ...
        progress.tick()          # increments by 1
        progress.tick("eval 2")  # increments and shows detail
        progress.finish()        # clears the line
    """

    def __init__(self, total: int, *, label: str = "", quiet: bool = False):
        self._total = total
        self._done = 0
        self._label = label
        self._quiet = quiet
        self._start = time.monotonic()
        self._last_line_len = 0

    def tick(self, detail: str = "") -> None:
        """Record one completed LLM call and update the display."""
        self._done += 1
        if self._quiet:
            return
        self._render(detail)

    def _render(self, detail: str = "") -> None:
        elapsed = time.monotonic() - self._start
        pct = self._done / self._total * 100 if self._total > 0 else 100

        if self._done > 0 and self._done < self._total:
            rate = elapsed / self._done
            remaining = rate * (self._total - self._done)
            eta = _format_duration(remaining)
            time_part = f"~{eta} remaining"
        elif self._done >= self._total:
            time_part = f"done in {_format_duration(elapsed)}"
        else:
            time_part = ""

        prefix = f"{self._label}: " if self._label else ""
        msg = f"{prefix}{pct:.0f}% ({self._done}/{self._total})"
        if detail:
            msg += f" {detail}"
        if time_part:
            msg += f" — {time_part}"

        # Overwrite the current line
        padded = msg[:120].ljust(self._last_line_len)
        sys.stderr.write(f"\r  {padded}")
        sys.stderr.flush()
        self._last_line_len = len(msg[:120])

    def finish(self) -> None:
        """Clear the progress line."""
        if self._quiet or not self._last_line_len:
            return
        sys.stderr.write(f"\r{' ' * (self._last_line_len + 4)}\r")
        sys.stderr.flush()
        self._last_line_len = 0

    @property
    def done(self) -> int:
        return self._done


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable short string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"
