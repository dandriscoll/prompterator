"""Tests for issue consolidation logic."""

import json

from tests.conftest import MockLLMClient

from prompterator.core.issue import (
    _determine_severity,
    _generate_issue_id,
    _split_feedback_entry,
    consolidate_feedback,
)
from prompterator.models.feedback import Feedback, FeedbackEntry


def test_consolidate_basic():
    """LLM is called and issues are created from its response."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="preamble at the top of the output")],
        ),
        Feedback(
            source_file="r2.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="conversational intro before the list")],
        ),
    ]

    llm_response = json.dumps([
        {
            "label": "preamble-insertion",
            "summary": "Output starts with unwanted conversational preamble",
            "evidence_indices": [0, 1],
        }
    ])
    mock = MockLLMClient(responses=[llm_response])

    result = consolidate_feedback(feedback_list, "test.prompt.txt", mock)
    assert len(result.issues) == 1
    assert result.issues[0].category == "preamble-insertion"
    assert result.issues[0].summary == "Output starts with unwanted conversational preamble"
    assert len(result.issues[0].evidence) == 2
    assert len(mock.calls) == 1


def test_consolidate_clusters():
    """LLM returns multiple clusters, each becomes a separate issue."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(text="preamble at the top"),
                FeedbackEntry(text="structural rewrite not requested"),
            ],
        ),
        Feedback(
            source_file="r2.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(text="chatty intro paragraph"),
                FeedbackEntry(text="replaced checkboxes with priority groups"),
            ],
        ),
    ]

    llm_response = json.dumps([
        {
            "label": "preamble-insertion",
            "summary": "Output starts with unwanted conversational preamble",
            "evidence_indices": [0, 2],
        },
        {
            "label": "structural-rewrite",
            "summary": "Model rewrites document structure instead of preserving it",
            "evidence_indices": [1, 3],
        },
    ])
    mock = MockLLMClient(responses=[llm_response])

    result = consolidate_feedback(feedback_list, "test.prompt.txt", mock)
    assert len(result.issues) == 2
    labels = [i.category for i in result.issues]
    assert "preamble-insertion" in labels
    assert "structural-rewrite" in labels


def test_min_occurrences_filter():
    """Clusters with too few unique sources are filtered by min_occurrences."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="preamble at the top")],
        ),
        Feedback(
            source_file="r2.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="output looks fine")],
        ),
    ]

    # Cluster with evidence from only r1 (1 unique source), min_occurrences=2
    llm_response = json.dumps([
        {
            "label": "preamble-insertion",
            "summary": "Output starts with unwanted preamble",
            "evidence_indices": [0],
        }
    ])
    mock = MockLLMClient(responses=[llm_response])

    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, min_occurrences=2
    )
    assert len(result.issues) == 0


def test_min_occurrences_skipped_with_single_source():
    """With only 1 feedback source, min_occurrences doesn't filter."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="preamble at the top")],
        ),
    ]

    llm_response = json.dumps([
        {
            "label": "preamble-insertion",
            "summary": "Output starts with unwanted preamble",
            "evidence_indices": [0],
        }
    ])
    mock = MockLLMClient(responses=[llm_response])

    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, min_occurrences=2
    )
    # Single source — threshold skipped, issue is kept
    assert len(result.issues) == 1


def test_severity_determination():
    """Severity is based on occurrence ratio."""
    assert _determine_severity(7, 10) == "high"   # 70%
    assert _determine_severity(5, 10) == "medium"  # 50%
    assert _determine_severity(2, 10) == "low"     # 20%
    assert _determine_severity(0, 0) == "medium"   # edge case


def test_issue_id_generation():
    """Issue IDs follow expected pattern (no category in ID)."""
    assert _generate_issue_id("test.prompt.txt", 1) == "issue-test-01"
    assert _generate_issue_id("foo.prompt.txt", 3) == "issue-foo-03"


# ---------------------------------------------------------------------------
# Feedback splitting
# ---------------------------------------------------------------------------

def test_split_semicolon():
    """Semicolon-separated items become independent observations."""
    parts = _split_feedback_entry("too chatty; incorrect grammar; missing examples")
    assert len(parts) == 3
    assert "too chatty" in parts[0]
    assert "incorrect grammar" in parts[1]
    assert "missing examples" in parts[2]


def test_split_no_semicolon():
    """Single observation stays as-is."""
    parts = _split_feedback_entry("the output adds a preamble before the list")
    assert parts == ["the output adds a preamble before the list"]


def test_split_short_parts_kept_together():
    """If splitting produces parts that are too short, keep original."""
    parts = _split_feedback_entry("ok; no")
    assert parts == ["ok; no"]


def test_split_one_meaningful_part():
    """If only one part is meaningful, keep original."""
    parts = _split_feedback_entry("the output is too verbose; ok")
    assert parts == ["the output is too verbose; ok"]


def test_split_preserves_long_text():
    """Long text without semicolons is unchanged."""
    text = "the output adds a conversational preamble and then proceeds to restructure the entire list"
    parts = _split_feedback_entry(text)
    assert parts == [text]
