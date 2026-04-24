"""Tests for the `prompterator check` command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from prompterator.commands.check import _extract_yaml_body, check_cmd


# ---------------------------------------------------------------------------
# _extract_yaml_body
# ---------------------------------------------------------------------------

def test_extract_frontmatter_strips_delimiters():
    """A file wrapped in --- delimiters has its body returned."""
    text = "---\nweights: 1,2,3\ncfg: 2.6\n---\n"
    body, was_fm = _extract_yaml_body(text)
    assert was_fm is True
    assert "---" not in body
    assert "weights" in body and "cfg" in body


def test_extract_plain_yaml_unchanged():
    """A plain YAML file (no ---) is returned as-is."""
    text = "weights: 1,2,3\ncfg: 2.6\n"
    body, was_fm = _extract_yaml_body(text)
    assert was_fm is False
    assert body == text


def test_extract_frontmatter_no_closing_still_works():
    """Opening --- without a matching close returns the rest; yaml can still parse it."""
    text = "---\nweights: 1,2,3\n"
    body, was_fm = _extract_yaml_body(text)
    assert was_fm is True
    assert "weights" in body


# ---------------------------------------------------------------------------
# check_cmd
# ---------------------------------------------------------------------------

def test_check_cmd_accepts_frontmatter_yaml(tmp_path: Path):
    """A valid frontmatter-wrapped YAML prompt passes check."""
    p = tmp_path / "lora.prompt.md"
    p.write_text(
        "---\n"
        "weights: 1,0,0,0.2,0,0,0.78,0.78,0.40,0.40,0.40,0.30\n"
        "cfg: 2.6\n"
        "trigger_weight: 1.1\n"
        "---\n"
    )
    result = CliRunner().invoke(check_cmd, [str(p)])
    assert result.exit_code == 0
    assert "OK" in result.output
    assert "frontmatter" in result.output
    assert "weights" in result.output


def test_check_cmd_accepts_plain_yaml(tmp_path: Path):
    """A plain YAML prompt (no ---) passes check."""
    p = tmp_path / "plain.prompt.md"
    p.write_text("weights: 1,2,3\ncfg: 2.6\n")
    result = CliRunner().invoke(check_cmd, [str(p)])
    assert result.exit_code == 0
    assert "plain YAML" in result.output


def test_check_cmd_rejects_malformed_yaml(tmp_path: Path):
    """Malformed YAML (e.g. unclosed bracket) exits 1."""
    p = tmp_path / "bad.prompt.md"
    p.write_text("---\nweights: [1, 2\n---\n")
    result = CliRunner().invoke(check_cmd, [str(p)])
    assert result.exit_code == 1
    assert "invalid YAML" in result.output


def test_check_cmd_rejects_non_mapping(tmp_path: Path):
    """A YAML document that's not a mapping (e.g. a bare list) is rejected."""
    p = tmp_path / "list.prompt.md"
    p.write_text("- one\n- two\n")
    result = CliRunner().invoke(check_cmd, [str(p)])
    assert result.exit_code == 1
    assert "expected a mapping" in result.output


def test_check_cmd_rejects_empty(tmp_path: Path):
    """An empty file (or one that parses to None) is rejected."""
    p = tmp_path / "empty.prompt.md"
    p.write_text("")
    result = CliRunner().invoke(check_cmd, [str(p)])
    assert result.exit_code == 1
    assert "empty" in result.output
