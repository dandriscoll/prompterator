"""Tests for feedback parsing, including prompt_ref extraction."""

import tempfile
from pathlib import Path

from prompterator.commands.feedback import (
    _extract_prior_prompt_ref,
    parse_mb_file,
)


# ── _extract_prior_prompt_ref tests ──


def test_extract_prior_prompt_ref_from_prompt_md():
    """Prompt .md file is found among multiple @prior lines."""
    content = """\
@source outputs/001-r1.out.md
@prior improve-todo.prompt.md
@prior 001.todoosy.md
<<< opens with a conversational paragraph
"""
    assert _extract_prior_prompt_ref(content) == "improve-todo.prompt.md"


def test_extract_prior_prompt_ref_from_prompt_txt():
    """Prompt .txt file is found."""
    content = """\
@source outputs/001-r1.out.md
@prior my-prompt.prompt.txt
@prior data.csv
<<< some feedback text
"""
    assert _extract_prior_prompt_ref(content) == "my-prompt.prompt.txt"


def test_extract_prior_prompt_ref_no_prompt():
    """Returns None when no @prior ends in .prompt.md/.prompt.txt."""
    content = """\
@source outputs/001-r1.out.md
@prior data.csv
<<< some feedback text
"""
    assert _extract_prior_prompt_ref(content) is None


def test_extract_prior_prompt_ref_first_match_wins():
    """If multiple prompt files exist, the first one wins."""
    content = """\
@prior first.prompt.md
@prior second.prompt.md
<<< some feedback text
"""
    assert _extract_prior_prompt_ref(content) == "first.prompt.md"


# ── parse_mb_file integration tests ──


def test_parse_mb_file_prompt_ref_from_prior():
    """Full .mb file uses @prior for prompt_ref, not @source."""
    content = """\
@source outputs/001-r1.out.md
@prior improve-todo.prompt.md
@prior 001.todoosy.md
<<< opens with a conversational paragraph that pollutes the output

---

@source outputs/001-r1.out.md
@prior improve-todo.prompt.md
@prior 001.todoosy.md
<<< replaced checkboxes with priority-grouped sections
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mb", delete=False, dir="/tmp"
    ) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        feedback = parse_mb_file(tmp_path)
        assert feedback.prompt_ref == "improve-todo.prompt.md"
        assert len(feedback.entries) == 2
        assert feedback.entries[0].text == "opens with a conversational paragraph that pollutes the output"
        assert feedback.entries[1].text == "replaced checkboxes with priority-grouped sections"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_parse_mb_file_all_entries_kept():
    """Feedback parsing keeps all entries including positive observations."""
    content = """\
@source outputs/001-r1.out.md
@prior test.prompt.md
<<< individual item rewrites are actually clearer than the originals
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mb", delete=False, dir="/tmp"
    ) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        feedback = parse_mb_file(tmp_path)
        assert len(feedback.entries) == 1
        assert feedback.entries[0].text == "individual item rewrites are actually clearer than the originals"
    finally:
        tmp_path.unlink(missing_ok=True)
