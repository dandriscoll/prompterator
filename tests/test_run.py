"""Tests for run directory creation."""

import re
from pathlib import Path

from prompterator.core.run import create_run_dir


def test_create_run_dir_creates_sequenced_dir(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path)
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    assert re.match(r"run001-\d{8}-\d{6}$", run_dir.name)


def test_create_run_dir_increments_sequence(tmp_path: Path) -> None:
    dir1 = create_run_dir(tmp_path)
    dir2 = create_run_dir(tmp_path)
    dir3 = create_run_dir(tmp_path)
    assert dir1.name.startswith("run001-")
    assert dir2.name.startswith("run002-")
    assert dir3.name.startswith("run003-")


def test_create_run_dir_creates_parents(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    run_dir = create_run_dir(nested)
    assert run_dir.exists()
    assert nested.exists()
