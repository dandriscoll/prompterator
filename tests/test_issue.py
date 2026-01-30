"""Tests for issue consolidation logic."""

from prompterator.core.issue import (
    _determine_severity,
    _generate_issue_id,
    consolidate_feedback,
)
from prompterator.models.feedback import Feedback, FeedbackEntry


def test_consolidate_single_category():
    """Consolidate feedback with only one category."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(category="clarity", value="unclear")],
        ),
    ]
    result = consolidate_feedback(feedback_list, "test.prompt.txt", ["clarity"])
    assert len(result.issues) == 1
    assert result.issues[0].category == "clarity"


def test_consolidate_multiple_categories(sample_feedback_list):
    """Consolidate feedback with multiple categories."""
    categories = ["clarity", "completeness", "tone"]
    result = consolidate_feedback(sample_feedback_list, "test.prompt.txt", categories)
    cats = [i.category for i in result.issues]
    assert "clarity" in cats
    assert "completeness" in cats
    assert "tone" in cats


def test_severity_determination():
    """Severity is based on occurrence ratio."""
    assert _determine_severity(7, 10) == "high"   # 70%
    assert _determine_severity(5, 10) == "medium"  # 50%
    assert _determine_severity(2, 10) == "low"     # 20%
    assert _determine_severity(0, 0) == "medium"   # edge case


def test_min_occurrences_filter():
    """Issues below min_occurrences are filtered out."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(category="clarity", value="ok")],
        ),
    ]
    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", ["clarity", "tone"], min_occurrences=2
    )
    assert len(result.issues) == 0


def test_issue_id_generation():
    """Issue IDs follow expected pattern."""
    assert _generate_issue_id("test.prompt.txt", "clarity", 1) == "issue-test-clarity-01"
    assert _generate_issue_id("foo.prompt.txt", "tone", 3) == "issue-foo-tone-03"
