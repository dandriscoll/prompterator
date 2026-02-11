"""Tests for eval specification generation."""

from prompterator.core.eval_spec import (
    CATEGORY_CRITERIA,
    _MAX_CRITERIA_PER_ISSUE,
    _deduplicate_details,
    _generate_eval_id,
    generate_evals_from_issues,
)
from prompterator.models.issue import Issue, IssueEvidence, IssueFile


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


# ── _deduplicate_details tests ──


def test_deduplicate_empty():
    """Empty list returns empty."""
    assert _deduplicate_details([]) == []


def test_deduplicate_no_overlap():
    """Distinct details are all kept."""
    details = [
        "output starts with conversational preamble",
        "changed checkbox format to bullet points",
        "chatbot sign-off at the end",
    ]
    result = _deduplicate_details(details)
    assert len(result) == 3


def test_deduplicate_synonyms():
    """Feedback using synonym words for the same theme is collapsed."""
    details = [
        "preamble at the top of the output",
        "conversational intro before the list",
        "chatty opening paragraph",
    ]
    result = _deduplicate_details(details)
    # All three describe the same "preamble" theme — only the longest kept.
    assert len(result) <= 2


def test_deduplicate_preserves_distinct_items():
    """All truly distinct details survive deduplication."""
    details = [
        "preamble at the top of output",
        "changed checkbox format to bullets",
        "dropped human context from items",
        "chatbot sign-off offering schedule",
    ]
    result = _deduplicate_details(details)
    assert len(result) == 4


def test_evidence_criteria_with_notes():
    """Criteria are derived from note= details when present."""
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-format-01",
                category="format",
                severity="high",
                summary="Format issues",
                evidence=[
                    IssueEvidence(source="r1.mb", feedback="format=bad; note=preamble before list"),
                    IssueEvidence(source="r2.mb", feedback="format=bad; note=chatbot sign-off at end"),
                    IssueEvidence(source="r3.mb", feedback="format=poor"),  # no detail
                ],
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file)
    criteria = eval_file.evals[0].rubric.criteria
    assert len(criteria) == 2
    assert any("preamble" in c for c in criteria)
    assert any("sign-off" in c for c in criteria)


def test_evidence_criteria_capped():
    """Even with many unique evidence details, criteria count is capped."""
    evidence = [
        IssueEvidence(source=f"r{i}.mb", feedback=f"format=bad; note=unique problem {i}")
        for i in range(20)
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-format-01",
                category="format",
                severity="high",
                summary="Format issues",
                evidence=evidence,
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file)
    assert len(eval_file.evals[0].rubric.criteria) <= _MAX_CRITERIA_PER_ISSUE
