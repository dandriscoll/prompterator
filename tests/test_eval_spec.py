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


def test_category_criteria_from_evidence(sample_issue_file):
    """Eval criteria come from evidence text since LLM-generated labels don't match CATEGORY_CRITERIA."""
    eval_file = generate_evals_from_issues(sample_issue_file)
    # First issue has evidence, so criteria come from evidence text
    first_eval = eval_file.evals[0]
    assert first_eval.rubric is not None
    assert all("Prompt addresses:" in c for c in first_eval.rubric.criteria)


def test_high_severity_all_required(sample_issue_file):
    """High severity issues use all_required scoring."""
    eval_file = generate_evals_from_issues(sample_issue_file)
    # First issue is high severity
    assert eval_file.evals[0].rubric.scoring == "all_required"
    # Second issue is low severity
    assert eval_file.evals[1].rubric.scoring == "any_required"


def test_eval_id_generation():
    """Eval IDs follow expected pattern."""
    assert _generate_eval_id("test.prompt.txt", "preamble-insertion", 1) == "eval-test-preamble-insertion-01"


def test_unknown_category_fallback():
    """LLM-generated labels that don't match CATEGORY_CRITERIA get a generic fallback criterion."""
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="preamble-insertion",
                severity="medium",
                summary="Output starts with unwanted preamble",
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file)
    assert len(eval_file.evals) == 1
    # No evidence → falls back to CATEGORY_CRITERIA lookup, which won't match → generic
    assert eval_file.evals[0].rubric.criteria == ["Addresses preamble-insertion concerns"]


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


def test_evidence_criteria_with_plain_text():
    """Criteria are derived from plain-text evidence feedback."""
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="preamble-insertion",
                severity="high",
                summary="Output starts with unwanted preamble",
                evidence=[
                    IssueEvidence(source="r1.mb", feedback="preamble before list"),
                    IssueEvidence(source="r2.mb", feedback="chatbot sign-off at end"),
                    IssueEvidence(source="r3.mb", feedback="changed checkbox format to bullets"),
                ],
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file)
    criteria = eval_file.evals[0].rubric.criteria
    assert len(criteria) == 3
    assert any("preamble" in c for c in criteria)
    assert any("sign-off" in c for c in criteria)
    assert any("checkbox" in c for c in criteria)


def test_evidence_criteria_capped():
    """Even with many unique evidence details, criteria count is capped."""
    evidence = [
        IssueEvidence(source=f"r{i}.mb", feedback=f"unique problem number {i}")
        for i in range(20)
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="preamble-insertion",
                severity="high",
                summary="Many problems",
                evidence=evidence,
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file)
    assert len(eval_file.evals[0].rubric.criteria) <= _MAX_CRITERIA_PER_ISSUE
