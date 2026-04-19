"""Tests for annotator core logic."""

from pathlib import Path

from prompterator.core.annotator import (
    build_editor_template,
    build_mb_block,
    build_mb_content,
    derive_mb_path,
    parse_editor_result,
)


# ---------------------------------------------------------------------------
# build_mb_block
# ---------------------------------------------------------------------------

def test_build_mb_block_single_input():
    result = build_mb_block("outputs/001.out.md", ["prompt.md"], "bad preamble")
    assert result == (
        "@file outputs/001.out.md\n"
        "@input prompt.md\n"
        "<<< bad preamble"
    )


def test_build_mb_block_multiple_inputs():
    result = build_mb_block(
        "outputs/001.out.md",
        ["improve-todo.prompt.md", "001.todoosy.md"],
        "replaced checkboxes",
    )
    assert "@input improve-todo.prompt.md\n" in result
    assert "@input 001.todoosy.md\n" in result
    assert result.endswith("<<< replaced checkboxes")


def test_build_mb_block_image_file():
    """Image files work the same as text files."""
    result = build_mb_block("outputs/001-r1.out.md", ["improve-todo.prompt.md"], "adds preamble")
    assert result.startswith("@file outputs/001-r1.out.md\n")


# ---------------------------------------------------------------------------
# build_mb_content
# ---------------------------------------------------------------------------

def test_build_mb_content_single_observation():
    content = build_mb_content("out.md", ["p.md"], ["one observation"])
    assert "---" not in content
    assert "<<< one observation" in content
    assert content.endswith("\n")


def test_build_mb_content_multiple_observations():
    content = build_mb_content(
        "outputs/001.out.md",
        ["prompt.md", "data.md"],
        ["first issue", "second issue", "third issue"],
    )
    blocks = content.split("\n---\n")
    assert len(blocks) == 3
    for block in blocks:
        assert "@file outputs/001.out.md" in block
        assert "@input prompt.md" in block
        assert "@input data.md" in block
    assert "<<< first issue" in blocks[0]
    assert "<<< second issue" in blocks[1]
    assert "<<< third issue" in blocks[2]


# ---------------------------------------------------------------------------
# derive_mb_path
# ---------------------------------------------------------------------------

def test_derive_mb_path_strips_directory_and_extensions():
    assert derive_mb_path("outputs/001-r1.out.md") == Path("001-r1.mb")


def test_derive_mb_path_simple_extension():
    assert derive_mb_path("001-r1.out.md") == Path("001-r1.mb")


def test_derive_mb_path_with_output_dir():
    result = derive_mb_path("outputs/001-r1.out.md", Path("feedback"))
    assert result == Path("feedback/001-r1.mb")


def test_derive_mb_path_no_extension():
    assert derive_mb_path("outputs/readme") == Path("readme.mb")


# ---------------------------------------------------------------------------
# build_editor_template / parse_editor_result
# ---------------------------------------------------------------------------

def test_editor_template_contains_context():
    template = build_editor_template("out.jpg", ["prompt.md", "blend.md"])
    assert "# File:  out.jpg" in template
    assert "# Input: prompt.md" in template
    assert "# Input: blend.md" in template


def test_parse_editor_result_strips_comments_and_blanks():
    text = (
        "# this is a comment\n"
        "first observation\n"
        "\n"
        "# another comment\n"
        "second observation\n"
        "\n"
    )
    obs = parse_editor_result(text)
    assert obs == ["first observation", "second observation"]


def test_parse_editor_result_empty():
    assert parse_editor_result("# only comments\n\n") == []


def test_parse_editor_result_preserves_content():
    text = "  pose not followed — model ignored constraint  \n"
    obs = parse_editor_result(text)
    assert obs == ["pose not followed — model ignored constraint"]


# ---------------------------------------------------------------------------
# round-trip: content -> parse by feedback command
# ---------------------------------------------------------------------------

def test_round_trip_parseable_by_markback(tmp_path):
    """Generated .mb content should be parseable by the feedback command."""
    content = build_mb_content(
        "outputs/001-r1.out.md",
        ["improve-todo.prompt.md", "001.todoosy.md"],
        ["preamble sentence", "replaced checkboxes"],
    )
    mb_file = tmp_path / "001-r1.mb"
    mb_file.write_text(content)

    # Parse it with the feedback command's parser
    from prompterator.commands.feedback import parse_mb_file

    feedback = parse_mb_file(mb_file)
    assert feedback.prompt_ref == "improve-todo.prompt.md"
    assert len(feedback.entries) == 2
    assert feedback.entries[0].text == "preamble sentence"
    assert feedback.entries[1].text == "replaced checkboxes"
