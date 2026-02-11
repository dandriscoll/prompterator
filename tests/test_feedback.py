"""Tests for feedback parsing, including prompt_ref extraction."""

import tempfile
from pathlib import Path

from prompterator.commands.feedback import (
    _extract_prior_prompt_ref,
    _parse_feedback_string,
    parse_mb_file,
)


# ── _extract_prior_prompt_ref tests ──


def test_extract_prior_prompt_ref_from_prompt_md():
    """Prompt .md file is found among multiple @prior lines."""
    content = """\
@source outputs/001-r1.out.md
@prior improve-todo.prompt.md
@prior 001.todoosy.md
<<< format=bad; note=preamble
"""
    assert _extract_prior_prompt_ref(content) == "improve-todo.prompt.md"


def test_extract_prior_prompt_ref_from_prompt_txt():
    """Prompt .txt file is found."""
    content = """\
@source outputs/001-r1.out.md
@prior my-prompt.prompt.txt
@prior data.csv
<<< clarity=low
"""
    assert _extract_prior_prompt_ref(content) == "my-prompt.prompt.txt"


def test_extract_prior_prompt_ref_no_prompt():
    """Returns None when no @prior ends in .prompt.md/.prompt.txt."""
    content = """\
@source outputs/001-r1.out.md
@prior data.csv
<<< clarity=low
"""
    assert _extract_prior_prompt_ref(content) is None


def test_extract_prior_prompt_ref_first_match_wins():
    """If multiple prompt files exist, the first one wins."""
    content = """\
@prior first.prompt.md
@prior second.prompt.md
<<< format=bad
"""
    assert _extract_prior_prompt_ref(content) == "first.prompt.md"


# ── _parse_feedback_string tests ──


def test_parse_feedback_simple():
    """Simple category=value parsing."""
    result = _parse_feedback_string("clarity=low")
    assert len(result) == 1
    assert result[0] == ("clarity", "low", None)


def test_parse_feedback_with_note():
    """Category with note modifier."""
    result = _parse_feedback_string("format=bad; note=preamble at top")
    assert len(result) == 1
    assert result[0] == ("format", "bad", "note=preamble at top")


def test_parse_feedback_multiple():
    """Multiple categories on one line."""
    result = _parse_feedback_string("clarity=low; completeness=missing")
    assert len(result) == 2


# ── parse_mb_file integration tests ──


def test_parse_mb_file_prompt_ref_from_prior():
    """Full .mb file uses @prior for prompt_ref, not @source."""
    content = """\
@source outputs/001-r1.out.md
@prior improve-todo.prompt.md
@prior 001.todoosy.md
<<< format=bad; note=preamble

---

@source outputs/001-r1.out.md
@prior improve-todo.prompt.md
@prior 001.todoosy.md
<<< accuracy=low; note=structural rewrite
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
    finally:
        tmp_path.unlink(missing_ok=True)


def test_parse_mb_file_positive_values_not_filtered():
    """Feedback parsing keeps all entries; filtering is done in issue consolidation."""
    content = """\
@source outputs/001-r1.out.md
@prior test.prompt.md
<<< clarity=good; note=items are clear
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mb", delete=False, dir="/tmp"
    ) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        feedback = parse_mb_file(tmp_path)
        assert len(feedback.entries) == 1
        assert feedback.entries[0].value == "good"
    finally:
        tmp_path.unlink(missing_ok=True)
