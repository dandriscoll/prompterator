"""Tests for prompt improvement logic."""

import json

from prompterator.core.improver import (
    _build_improvement_prompt,
    _parse_improvement_response,
    generate_improved_prompt_with_rationale,
)
from prompterator.models.result import EvalResult

from tests.conftest import MockLLMClient


def test_build_improvement_prompt_structure(sample_issue_file):
    """Improvement prompt includes issues and instructions."""
    prompt = _build_improvement_prompt("Original text", sample_issue_file)
    assert "Original text" in prompt
    assert "unclear-instructions" in prompt
    assert "missing-examples" in prompt
    assert "ALL" in prompt or "all failing issues" in prompt.lower()


def test_surgical_prompt_includes_rationale(sample_issue_file):
    """Prompt requests JSON with rationale field."""
    prompt = _build_improvement_prompt("Original text", sample_issue_file)
    assert "rationale" in prompt
    assert "improved_prompt" in prompt


def test_build_improvement_prompt_with_eval_results(sample_issue_file):
    """Eval results are included when provided."""
    results = [
        EvalResult(eval_id="eval-01", passed=False, score=0.33, details="Needs work"),
    ]
    prompt = _build_improvement_prompt("Original", sample_issue_file, eval_results=results, iteration=2)
    assert "eval-01" in prompt
    assert "ITERATION: 2" in prompt


def test_parse_improvement_response_json():
    """Valid JSON response is parsed correctly."""
    response = json.dumps({
        "rationale": "Clarified instructions",
        "changed_section": "Opening paragraph",
        "improved_prompt": "Better prompt text",
    })
    rationale, improved = _parse_improvement_response(response)
    assert rationale == "Clarified instructions"
    assert improved == "Better prompt text"


def test_parse_improvement_response_fallback():
    """Non-JSON response falls back to treating it as the prompt."""
    response = "Just a plain improved prompt"
    rationale, improved = _parse_improvement_response(response)
    assert improved == "Just a plain improved prompt"


def test_generate_improved_prompt_with_rationale(sample_issue_file):
    """Full generation returns improved text, rationale, and raw output."""
    llm_response = json.dumps({
        "rationale": "Added examples",
        "changed_section": "Body",
        "improved_prompt": "Improved prompt with examples",
    })
    llm = MockLLMClient(responses=[llm_response])

    improved, rationale, raw = generate_improved_prompt_with_rationale(
        "Original prompt", sample_issue_file, llm, iteration=1
    )
    assert improved == "Improved prompt with examples"
    assert rationale == "Added examples"
    assert raw == llm_response
