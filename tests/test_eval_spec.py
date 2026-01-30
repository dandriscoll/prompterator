"""Tests for eval specification generation."""

from prompterator.core.eval_spec import (
    CATEGORY_CRITERIA,
    _generate_eval_id,
    generate_evals_from_issues,
)
from prompterator.models.issue import Issue, IssueFile


def test_generate_evals_from_issues(sample_issue_file):
    """Evals are generated for each issue."""
    eval_file = generate_evals_from_issues(sample_issue_file)
    assert len(eval_file.evals) == len(sample_issue_file.issues)
    assert eval_file.prompt_ref == sample_issue_file.prompt_ref


def test_category_criteria_mapping(sample_issue_file):
    """Eval criteria come from CATEGORY_CRITERIA mapping."""
    eval_file = generate_evals_from_issues(sample_issue_file)
    clarity_eval = eval_file.evals[0]
    assert clarity_eval.rubric is not None
    assert clarity_eval.rubric.criteria == CATEGORY_CRITERIA["clarity"]


def test_high_severity_all_required(sample_issue_file):
    """High severity issues use all_required scoring."""
    eval_file = generate_evals_from_issues(sample_issue_file)
    # First issue is high severity
    assert eval_file.evals[0].rubric.scoring == "all_required"
    # Second issue is low severity
    assert eval_file.evals[1].rubric.scoring == "any_required"


def test_eval_id_generation():
    """Eval IDs follow expected pattern."""
    assert _generate_eval_id("test.prompt.txt", "clarity", 1) == "eval-test-clarity-01"


def test_unknown_category_fallback():
    """Unknown categories get a generic criterion."""
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-custom-01",
                category="custom_thing",
                severity="medium",
                summary="Custom issue",
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file)
    assert len(eval_file.evals) == 1
    assert eval_file.evals[0].rubric.criteria == ["Addresses custom_thing concerns"]
