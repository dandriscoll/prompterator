"""Tests for eval specification generation."""

import json

from prompterator.core.eval_spec import (
    CATEGORY_CRITERIA,
    _MAX_CRITERIA_PER_ISSUE,
    _deduplicate_details,
    _generate_eval_id,
    _reconcile_evals_with_issues,
    generate_evals_from_issues,
)
from prompterator.models.eval import Eval, EvalRubric
from prompterator.models.issue import Issue, IssueEvidence, IssueFile
from prompterator.runners.llm import LLMClient

from tests.conftest import MockLLMClient


def test_generate_evals_from_issues(sample_issue_file):
    """Evals are generated for each issue."""
    eval_file = generate_evals_from_issues(sample_issue_file)
    assert len(eval_file.evals) == len(sample_issue_file.issues)
    assert eval_file.prompt_ref == sample_issue_file.prompt_ref


def test_fallback_criteria_from_evidence(sample_issue_file):
    """Without LLM, eval criteria are derived from evidence text."""
    eval_file = generate_evals_from_issues(sample_issue_file)
    first_eval = eval_file.evals[0]
    assert first_eval.rubric is not None
    # Without LLM, falls back to evidence-derived criteria
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


def test_llm_criteria_inversion():
    """With LLM, a single criterion is produced by inverting the issue."""
    from unittest.mock import MagicMock

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = '["Output begins directly with the requested content"]'

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
                ],
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file, llm_client=mock_llm)
    criteria = eval_file.evals[0].rubric.criteria
    assert len(criteria) == 1
    assert criteria[0] == "Output begins directly with the requested content"
    # LLM receives only the issue summary, not evidence
    mock_llm.generate.assert_called_once_with(
        "Output starts with unwanted preamble", system=mock_llm.generate.call_args.kwargs["system"]
    )


def test_evidence_fallback_without_llm():
    """Without LLM, falls back to evidence-derived criteria."""
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


def test_llm_returns_single_criterion_even_if_many_returned():
    """Only one criterion is kept even if LLM returns multiple."""
    from unittest.mock import MagicMock

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = json.dumps(
        ["First criterion", "Second criterion", "Third criterion"]
    )

    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="preamble-insertion",
                severity="high",
                summary="Some problem",
            ),
        ],
    )
    eval_file = generate_evals_from_issues(issue_file, llm_client=mock_llm)
    assert len(eval_file.evals[0].rubric.criteria) == 1


# ---------------------------------------------------------------------------
# Issue reorganization / eval reconciliation
# ---------------------------------------------------------------------------

def _make_eval(eval_id, issue_ref, criteria):
    return Eval(
        id=eval_id,
        type="rubric",
        issue_ref=issue_ref,
        description=f"Check {eval_id}",
        rubric=EvalRubric(criteria=criteria),
    )


def test_reconcile_maps_by_meaning():
    """LLM maps existing evals to new issues by semantic match."""
    existing_evals = [
        _make_eval("eval-old-01", "issue-old-01", ["Output does not add preamble"]),
        _make_eval("eval-old-02", "issue-old-02", ["Output preserves checkbox format"]),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(id="issue-new-A", category="preamble", severity="high",
                  summary="Output adds conversational preamble"),
            Issue(id="issue-new-B", category="formatting", severity="medium",
                  summary="Output changes checkbox format"),
        ],
    )

    llm = MockLLMClient(responses=[json.dumps([
        {"eval_id": "eval-old-01", "action": "keep", "new_issue_id": "issue-new-A"},
        {"eval_id": "eval-old-02", "action": "keep", "new_issue_id": "issue-new-B"},
    ])])

    mapping = _reconcile_evals_with_issues(existing_evals, issue_file, llm)
    assert mapping == {"eval-old-01": "issue-new-A", "eval-old-02": "issue-new-B"}


def test_reconcile_drops_obsolete_evals():
    """Evals for removed issues are dropped."""
    existing_evals = [
        _make_eval("eval-old-01", "issue-old-01", ["Output does not add preamble"]),
        _make_eval("eval-old-02", "issue-old-02", ["Output preserves tone"]),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(id="issue-new-A", category="preamble", severity="high",
                  summary="Preamble problem"),
        ],
    )

    llm = MockLLMClient(responses=[json.dumps([
        {"eval_id": "eval-old-01", "action": "keep", "new_issue_id": "issue-new-A"},
        {"eval_id": "eval-old-02", "action": "drop"},
    ])])

    mapping = _reconcile_evals_with_issues(existing_evals, issue_file, llm)
    assert "eval-old-01" in mapping
    assert "eval-old-02" not in mapping


def test_generate_evals_preserves_criteria_after_reorg():
    """Hand-tuned criteria survive issue reorganization."""
    existing_evals = [
        _make_eval("eval-old-01", "issue-old-01", ["My hand-tuned criterion"]),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(id="issue-new-X", category="preamble", severity="high",
                  summary="Preamble problem"),
        ],
    )

    # First call: reconciliation, second call: would be criteria generation (shouldn't happen)
    llm = MockLLMClient(responses=[
        json.dumps([
            {"eval_id": "eval-old-01", "action": "keep", "new_issue_id": "issue-new-X"},
        ]),
    ])

    eval_file = generate_evals_from_issues(issue_file, llm, existing_evals=existing_evals)
    assert len(eval_file.evals) == 1
    ev = eval_file.evals[0]
    assert ev.issue_ref == "issue-new-X"  # updated to new ID
    assert ev.rubric.criteria == ["My hand-tuned criterion"]  # preserved


def test_generate_evals_creates_new_for_unmatched_issues():
    """New issues without matching evals get fresh criteria."""
    existing_evals = [
        _make_eval("eval-old-01", "issue-old-01", ["Old criterion"]),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(id="issue-new-X", category="preamble", severity="high",
                  summary="Preamble problem"),
            Issue(id="issue-new-Y", category="formatting", severity="medium",
                  summary="Format problem",
                  evidence=[IssueEvidence(source="r1.mb", feedback="format issue")]),
        ],
    )

    llm = MockLLMClient(responses=[
        # reconciliation: old eval maps to issue-new-X
        json.dumps([
            {"eval_id": "eval-old-01", "action": "keep", "new_issue_id": "issue-new-X"},
        ]),
        # criteria generation for issue-new-Y
        '["Output preserves original formatting"]',
    ])

    eval_file = generate_evals_from_issues(issue_file, llm, existing_evals=existing_evals)
    assert len(eval_file.evals) == 2
    assert eval_file.evals[0].rubric.criteria == ["Old criterion"]  # preserved
    assert eval_file.evals[1].rubric.criteria == ["Output preserves original formatting"]  # new


def test_generate_evals_no_reorg_skips_reconciliation():
    """When issue IDs match, no reconciliation LLM call is made."""
    existing_evals = [
        _make_eval("eval-01", "issue-01", ["Existing criterion"]),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(id="issue-01", category="preamble", severity="high",
                  summary="Same issue"),
        ],
    )

    llm = MockLLMClient()
    eval_file = generate_evals_from_issues(issue_file, llm, existing_evals=existing_evals)

    assert len(eval_file.evals) == 1
    assert eval_file.evals[0].rubric.criteria == ["Existing criterion"]
    assert len(llm.calls) == 0  # no LLM calls — direct ID match
