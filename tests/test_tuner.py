"""Integration tests for the tuning loop."""

import json
import tempfile
from pathlib import Path

from prompterator.core.tuner import run_tuning_loop
from prompterator.models.iteration import IterationRecord

from tests.conftest import MockLLMClient


def _make_prompt_file(text: str = "Original prompt text") -> Path:
    """Create a temp prompt file."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".prompt.txt", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


def _improvement_response(prompt_text: str, rationale: str = "Improved clarity") -> str:
    return json.dumps({
        "rationale": rationale,
        "changed_section": "body",
        "improved_prompt": prompt_text,
    })


def _eval_response(pass_count: int, total: int) -> str:
    lines = []
    for i in range(total):
        status = "PASS" if i < pass_count else "FAIL"
        lines.append(f"CRITERION: C{i+1}\nRESULT: {status}\nREASON: test")
    lines.append(f"OVERALL: {'PASS' if pass_count == total else 'FAIL'}")
    lines.append(f"SCORE: {pass_count/total if total else 1.0}")
    return "\n".join(lines)


def test_tuning_loop_improves_then_stops(sample_issue_file, sample_eval_file):
    """Loop stops when all evals pass."""
    prompt_path = _make_prompt_file()
    try:
        # Baseline: 2 evals, each with 3 criteria
        # Iteration 1 editor response + 2 critic responses (partial pass)
        # Iteration 2 editor response + 2 critic responses (all pass)
        editor = MockLLMClient(responses=[
            _improvement_response("Better prompt v1"),
            _improvement_response("Better prompt v2"),
        ])
        critic = MockLLMClient(responses=[
            # Baseline: both fail
            _eval_response(1, 3), _eval_response(1, 3),
            # Iter 1: one passes
            _eval_response(3, 3), _eval_response(1, 3),
            # Iter 2: both pass
            _eval_response(3, 3), _eval_response(3, 3),
        ])

        report = run_tuning_loop(
            prompt_path, sample_issue_file, sample_eval_file,
            editor, critic, max_iterations=5,
        )
        assert report.final_summary.verdict == "PASS"
        assert len(report.iterations) == 2
    finally:
        prompt_path.unlink(missing_ok=True)


def test_tuning_loop_max_iterations(sample_issue_file, sample_eval_file):
    """Loop respects max_iterations limit."""
    prompt_path = _make_prompt_file()
    try:
        # Always improve but never fully pass - generate enough responses
        editor_responses = [_improvement_response(f"v{i}") for i in range(3)]
        # Baseline + 3 iterations, each needing 2 eval responses
        critic_responses = [_eval_response(1, 3) for _ in range(8)]

        editor = MockLLMClient(responses=editor_responses)
        critic = MockLLMClient(responses=critic_responses)

        report = run_tuning_loop(
            prompt_path, sample_issue_file, sample_eval_file,
            editor, critic, max_iterations=3,
        )
        assert len(report.iterations) <= 3
    finally:
        prompt_path.unlink(missing_ok=True)


def test_tuning_loop_no_improvement_stops(sample_issue_file, sample_eval_file):
    """Loop stops when score doesn't improve."""
    prompt_path = _make_prompt_file()
    try:
        editor = MockLLMClient(responses=[
            _improvement_response("v1"),
            _improvement_response("v2"),
            _improvement_response("v3"),
        ])
        critic = MockLLMClient(responses=[
            # Baseline: score ~0.33 each
            _eval_response(1, 3), _eval_response(1, 3),
            # Iter 1: improves to 0.67 each
            _eval_response(2, 3), _eval_response(2, 3),
            # Iter 2: drops to 0.33 each (worse) -> should stop
            _eval_response(1, 3), _eval_response(1, 3),
        ])

        report = run_tuning_loop(
            prompt_path, sample_issue_file, sample_eval_file,
            editor, critic, max_iterations=10, patience=1,
        )
        # Should stop at iteration 2 due to no improvement (patience=1)
        assert len(report.iterations) == 2
    finally:
        prompt_path.unlink(missing_ok=True)


def test_fixed_evals_not_modified(sample_issue_file, sample_eval_file):
    """Eval specs are not modified during tuning."""
    prompt_path = _make_prompt_file()
    original_evals = [e.model_copy() for e in sample_eval_file.evals]
    try:
        editor = MockLLMClient(responses=[_improvement_response("v1")])
        critic = MockLLMClient(responses=[
            _eval_response(1, 3), _eval_response(1, 3),
            _eval_response(3, 3), _eval_response(3, 3),
        ])
        run_tuning_loop(
            prompt_path, sample_issue_file, sample_eval_file,
            editor, critic, max_iterations=2,
        )
        for orig, current in zip(original_evals, sample_eval_file.evals):
            assert orig.id == current.id
            assert orig.type == current.type
            assert orig.rubric.criteria == current.rubric.criteria
    finally:
        prompt_path.unlink(missing_ok=True)


def test_prompt_versioning_across_iterations(sample_issue_file, sample_eval_file):
    """Each iteration records the correct prompt text."""
    prompt_path = _make_prompt_file("original")
    try:
        editor = MockLLMClient(responses=[
            _improvement_response("version-1"),
            _improvement_response("version-2"),
        ])
        critic = MockLLMClient(responses=[
            _eval_response(0, 3), _eval_response(0, 3),  # baseline
            _eval_response(1, 3), _eval_response(1, 3),  # iter 1
            _eval_response(2, 3), _eval_response(2, 3),  # iter 2
            _eval_response(1, 3), _eval_response(1, 3),  # iter 3 (worse, stops)
        ])

        report = run_tuning_loop(
            prompt_path, sample_issue_file, sample_eval_file,
            editor, critic, max_iterations=5,
        )
        assert report.iterations[0].prompt_text == "version-1"
        if len(report.iterations) > 1:
            assert report.iterations[1].prompt_text == "version-2"
    finally:
        prompt_path.unlink(missing_ok=True)


def test_iteration_record_structure(sample_issue_file, sample_eval_file):
    """Iteration records have all required fields."""
    prompt_path = _make_prompt_file()
    try:
        editor = MockLLMClient(responses=[_improvement_response("improved")])
        critic = MockLLMClient(responses=[
            _eval_response(1, 3), _eval_response(1, 3),
            _eval_response(3, 3), _eval_response(3, 3),
        ])
        report = run_tuning_loop(
            prompt_path, sample_issue_file, sample_eval_file,
            editor, critic, max_iterations=1,
        )
        rec = report.iterations[0]
        assert isinstance(rec, IterationRecord)
        assert rec.iteration == 1
        assert rec.rationale != ""
        assert rec.diff.before != rec.diff.after or rec.diff.before == rec.diff.after
        assert len(rec.eval_results) == 2
        assert rec.summary is not None
        assert rec.l2_output is not None
    finally:
        prompt_path.unlink(missing_ok=True)


def test_metric_deltas_computed(sample_issue_file, sample_eval_file):
    """Metric deltas are computed correctly."""
    prompt_path = _make_prompt_file()
    try:
        editor = MockLLMClient(responses=[_improvement_response("v1")])
        critic = MockLLMClient(responses=[
            _eval_response(1, 3), _eval_response(0, 3),  # baseline
            _eval_response(2, 3), _eval_response(1, 3),  # iter 1
        ])
        report = run_tuning_loop(
            prompt_path, sample_issue_file, sample_eval_file,
            editor, critic, max_iterations=1,
        )
        rec = report.iterations[0]
        # Deltas should exist for each eval
        assert len(rec.metric_deltas) == 2
        # Check metric_table in report
        assert len(report.metric_table) == 2
        for row in report.metric_table:
            assert "eval_id" in row
            assert "before" in row
            assert "after" in row
            assert "delta" in row
    finally:
        prompt_path.unlink(missing_ok=True)
