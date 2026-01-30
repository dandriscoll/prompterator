"""Tests for eval runner and metric computation."""

import tempfile
from pathlib import Path

from prompterator.core.eval_runner import (
    _build_rubric_prompt,
    _parse_rubric_response,
    run_all_evals,
    run_eval,
)
from prompterator.models.eval import Eval, EvalRubric

from tests.conftest import MockLLMClient


def test_parse_rubric_response_all_pass():
    """All criteria passing gives score 1.0."""
    response = """CRITERION: Clear
RESULT: PASS
REASON: Very clear

CRITERION: Precise
RESULT: PASS
REASON: Precise language

OVERALL: PASS
SCORE: 1.0"""
    passed, score, details = _parse_rubric_response(response, ["Clear", "Precise"])
    assert passed is True
    assert score == 1.0


def test_parse_rubric_response_mixed():
    """Mixed results give partial score."""
    response = """CRITERION: Clear
RESULT: PASS
REASON: Good clarity

CRITERION: Precise
RESULT: FAIL
REASON: Vague in places

CRITERION: Examples
RESULT: FAIL
REASON: No examples

OVERALL: FAIL
SCORE: 0.33"""
    passed, score, details = _parse_rubric_response(response, ["Clear", "Precise", "Examples"])
    assert passed is False
    assert abs(score - 1 / 3) < 0.01


def test_build_rubric_prompt():
    """Rubric prompt includes criteria and prompt content."""
    prompt = _build_rubric_prompt("Test prompt", ["Criterion A", "Criterion B"])
    assert "Criterion A" in prompt
    assert "Criterion B" in prompt
    assert "Test prompt" in prompt


def test_run_eval_rubric():
    """Rubric eval returns correct result from mocked LLM."""
    llm = MockLLMClient(responses=[
        "CRITERION: Clear\nRESULT: PASS\nREASON: Good\nOVERALL: PASS\nSCORE: 1.0"
    ])
    eval_spec = Eval(
        id="eval-01",
        type="rubric",
        rubric=EvalRubric(criteria=["Clear"]),
    )
    result = run_eval(eval_spec, "Test prompt", llm)
    assert result.passed is True
    assert result.score == 1.0
    assert result.eval_id == "eval-01"


def test_run_eval_assertion():
    """Assertion eval returns correct result from mocked LLM."""
    llm = MockLLMClient(responses=[
        "RESULT: PASS\nREASON: Assertion satisfied"
    ])
    eval_spec = Eval(
        id="eval-02",
        type="assertion",
        assertion="Prompt mentions examples",
    )
    result = run_eval(eval_spec, "Test prompt with examples", llm)
    assert result.passed is True
    assert result.score == 1.0


def test_result_summary_calculation(sample_eval_file):
    """Summary correctly aggregates results."""
    # Two evals: first fails, second passes
    llm = MockLLMClient(responses=[
        "CRITERION: A\nRESULT: FAIL\nREASON: Bad\nCRITERION: B\nRESULT: FAIL\nREASON: Bad\nCRITERION: C\nRESULT: FAIL\nREASON: Bad\nOVERALL: FAIL\nSCORE: 0.0",
        "CRITERION: A\nRESULT: PASS\nREASON: Good\nCRITERION: B\nRESULT: PASS\nREASON: Good\nCRITERION: C\nRESULT: PASS\nREASON: Good\nOVERALL: PASS\nSCORE: 1.0",
    ])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test prompt")
        tmp = Path(f.name)
    try:
        result_file = run_all_evals(sample_eval_file, tmp, llm)
    finally:
        tmp.unlink()

    assert result_file.summary.verdict == "PARTIAL"
    assert result_file.summary.passed_count == 1
    assert result_file.summary.failed_count == 1
    assert 0.0 < result_file.summary.overall_score < 1.0
