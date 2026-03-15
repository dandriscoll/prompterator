"""Tests for prompt improvement logic."""

from prompterator.core.improver import (
    _apply_edit,
    _build_edit_prompt,
    generate_improved_prompt_with_rationale,
)
from prompterator.models.result import EvalResult

from tests.conftest import MockLLMClient


def test_build_edit_prompt_structure(sample_issue_file):
    """Edit prompt includes the prompt text and issues."""
    prompt = _build_edit_prompt("Original text", sample_issue_file)
    assert "Original text" in prompt
    assert "unclear-instructions" in prompt
    assert "missing-examples" in prompt


def test_build_edit_prompt_with_eval_results(sample_issue_file):
    """Eval results are included when provided."""
    results = [
        EvalResult(eval_id="eval-01", passed=False, score=0.33, details="Needs work"),
    ]
    prompt = _build_edit_prompt("Original", sample_issue_file, eval_results=results, iteration=2)
    assert "eval-01" in prompt
    assert "ITERATION: 2" in prompt


def test_apply_edit_replace():
    """REPLACE action finds and replaces text."""
    original = "You are a helpful assistant.\n\nBe concise."
    response = (
        "RATIONALE: Add preamble prohibition\n"
        "ACTION: REPLACE\n"
        "FIND: Be concise.\n"
        "REPLACE_WITH: Be concise. Do not add conversational preamble."
    )
    edited, rationale = _apply_edit(original, response)
    assert "Do not add conversational preamble" in edited
    assert "Be concise" in edited
    assert "Add preamble prohibition" in rationale


def test_apply_edit_append():
    """APPEND action adds text at the end."""
    original = "You are a helpful assistant."
    response = (
        "RATIONALE: Add constraint\n"
        "ACTION: APPEND\n"
        "APPEND_TEXT: Never use emojis."
    )
    edited, rationale = _apply_edit(original, response)
    assert edited.endswith("Never use emojis.")
    assert "You are a helpful assistant." in edited


def test_apply_edit_prepend():
    """PREPEND action adds text at the beginning."""
    original = "Be concise."
    response = (
        "RATIONALE: Add role\n"
        "ACTION: PREPEND\n"
        "PREPEND_TEXT: You are an expert editor."
    )
    edited, rationale = _apply_edit(original, response)
    assert edited.startswith("You are an expert editor.")
    assert "Be concise." in edited


def test_apply_edit_find_not_matched():
    """Falls back to append when FIND text doesn't match."""
    original = "You are a helpful assistant."
    response = (
        "RATIONALE: Fix something\n"
        "ACTION: REPLACE\n"
        "FIND: This text does not exist in the prompt at all\n"
        "REPLACE_WITH: New text here"
    )
    edited, rationale = _apply_edit(original, response)
    assert "New text here" in edited
    assert "appended instead" in rationale


def test_generate_structured_edit(sample_issue_file):
    """Full generation with structured edit."""
    llm_response = (
        "RATIONALE: Prohibit preamble to pass eval\n"
        "ACTION: APPEND\n"
        "APPEND_TEXT: Do not add conversational preamble before the list."
    )
    llm = MockLLMClient(responses=[llm_response])

    improved, rationale, raw = generate_improved_prompt_with_rationale(
        "You are a helpful assistant.", sample_issue_file, llm, iteration=1
    )
    assert "Do not add conversational preamble before the list." in improved
    assert "You are a helpful assistant." in improved
    assert "Prohibit preamble" in rationale
