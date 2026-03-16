"""Tests for Progress counter."""

from prompterator.core.progress import Progress, _format_duration


def test_format_duration_seconds():
    assert _format_duration(5) == "5s"
    assert _format_duration(59) == "59s"


def test_format_duration_minutes():
    assert _format_duration(60) == "1m00s"
    assert _format_duration(90) == "1m30s"
    assert _format_duration(3599) == "59m59s"


def test_format_duration_hours():
    assert _format_duration(3600) == "1h00m"
    assert _format_duration(3661) == "1h01m"


def test_progress_tick_increments():
    p = Progress(10, quiet=True)
    assert p.done == 0
    p.tick()
    assert p.done == 1
    p.tick()
    assert p.done == 2


def test_progress_quiet_no_output(capsys):
    p = Progress(5, quiet=True)
    for _ in range(5):
        p.tick("detail")
    p.finish()
    captured = capsys.readouterr()
    assert captured.err == ""


def test_progress_finish_clears(capsys):
    """Non-quiet progress writes to stderr then clears on finish."""
    p = Progress(2, label="Test")
    p.tick()
    p.finish()
    captured = capsys.readouterr()
    # After finish, the last write should be a clearing line
    assert captured.err.endswith("\r")
