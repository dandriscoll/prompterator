"""Tests for issue consolidation logic."""

import json

from tests.conftest import MockLLMClient

from prompterator.core.issue import (
    _CLUSTER_SYSTEM,
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


def test_cluster_prompt_documents_parens_convention():
    """Tripwire: clustering prompt must still describe the `Category (instance)` convention."""
    assert "Category (specific instance)" in _CLUSTER_SYSTEM
    assert "parenthesised body" in _CLUSTER_SYSTEM


def test_cluster_prompt_documents_themes():
    """Tripwire: clustering prompt must describe themes as authoritative + proposal mode."""
    assert "THEMES ARE AUTHORITATIVE" in _CLUSTER_SYSTEM
    assert "unassigned" in _CLUSTER_SYSTEM
    assert "PROPOSAL MODE" in _CLUSTER_SYSTEM
    for key in ("anchors", "clusters", "analysis", "missing_themes", "theme_adjustments"):
        assert key in _CLUSTER_SYSTEM


# ---------------------------------------------------------------------------
# Themes: authoritative constraint + proposal mode + anchor plumbing
# ---------------------------------------------------------------------------

def _themed_response(anchors, clusters, analysis=None):
    return json.dumps({
        "anchors": anchors,
        "clusters": clusters,
        "analysis": analysis or {"missing_themes": [], "theme_adjustments": []},
    })


def test_themes_constrain_cluster_categories():
    """When themes is set, clusters with off-vocabulary themes are dropped and counted."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="preamble at the top of the output")],
        ),
        Feedback(
            source_file="r2.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="invented-category thing")],
        ),
    ]
    response = _themed_response(
        anchors=[
            {"index": 0, "instance": "conversational preamble", "confidence": "high", "themes": ["tone"]},
            {"index": 1, "instance": "some other thing", "confidence": "medium", "themes": ["made-up"]},
        ],
        clusters=[
            {"theme": "tone", "failure_mode": "chatty-preamble", "summary": "preamble", "anchor_indices": [0]},
            {"theme": "made-up", "failure_mode": "bogus", "summary": "bogus", "anchor_indices": [1]},
        ],
    )
    mock = MockLLMClient(responses=[response])

    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, themes=["tone", "structure"]
    )

    # Invented theme dropped, valid theme kept
    assert len(result.issues) == 1
    assert result.issues[0].category == "tone"
    assert result.analysis.dropped_invented_theme_count == 1


def test_themes_proposal_mode_when_empty():
    """Empty themes list — LLM's proposed themes pass through and land in analysis."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="preamble at the top")],
        ),
    ]
    response = _themed_response(
        anchors=[{"index": 0, "instance": "preamble", "confidence": "high", "themes": ["tone"]}],
        clusters=[
            {"theme": "tone", "failure_mode": "chatty-preamble", "summary": "preamble", "anchor_indices": [0]}
        ],
        analysis={"missing_themes": ["tone"], "theme_adjustments": []},
    )
    mock = MockLLMClient(responses=[response])

    result = consolidate_feedback(feedback_list, "test.prompt.txt", mock, themes=[])

    assert len(result.issues) == 1
    # Proposal-mode themes are kept as-is (no drop), and surfaced in analysis
    assert result.analysis.dropped_invented_theme_count == 0
    assert "tone" in result.analysis.missing_themes


def test_anchor_fields_flow_to_issue_evidence():
    """instance + confidence from the anchor land on IssueEvidence."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="Style issue (rushed cadence)")],
        ),
    ]
    response = _themed_response(
        anchors=[{"index": 0, "instance": "rushed cadence", "confidence": "high", "themes": ["style"]}],
        clusters=[
            {"theme": "style", "failure_mode": "rushed-cadence", "summary": "cadence feels rushed",
             "anchor_indices": [0]}
        ],
    )
    mock = MockLLMClient(responses=[response])

    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, themes=["style"]
    )
    assert len(result.issues) == 1
    ev = result.issues[0].evidence[0]
    assert ev.instance == "rushed cadence"
    assert ev.confidence == "high"
    # raw feedback preserved alongside the distilled anchor
    assert "rushed cadence" in ev.feedback


def test_anchor_in_multiple_themes_produces_multiple_issues():
    """An anchor that supports two themes appears as evidence under both."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="rambling tone; unclear structure")],
        ),
    ]
    # The feedback gets split on ';' into two observations; for this test we
    # let the LLM return one anchor per observation and put anchor 0 under
    # both 'tone' and 'structure' clusters — the same underlying problem
    # surfaces in two issues.
    response = _themed_response(
        anchors=[
            {"index": 0, "instance": "rambling tone", "confidence": "medium", "themes": ["tone", "structure"]},
            {"index": 1, "instance": "unclear structure", "confidence": "medium", "themes": ["structure"]},
        ],
        clusters=[
            {"theme": "tone", "failure_mode": "rambling", "summary": "rambling", "anchor_indices": [0]},
            {"theme": "structure", "failure_mode": "unclear", "summary": "unclear", "anchor_indices": [0, 1]},
        ],
    )
    mock = MockLLMClient(responses=[response])
    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, themes=["tone", "structure"]
    )

    by_theme = {i.category: i for i in result.issues}
    assert set(by_theme) == {"tone", "structure"}
    # anchor 0 appears in both issues (duplication across themes is intended)
    assert any("rambling" in (ev.instance or "") for ev in by_theme["tone"].evidence)
    assert any("rambling" in (ev.instance or "") for ev in by_theme["structure"].evidence)


def test_legacy_response_shape_still_parsed():
    """Back-compat: a bare array of clusters (pre-themes shape) still produces issues."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="preamble at the top")],
        ),
    ]
    legacy = json.dumps([
        {"label": "chatty-preamble", "summary": "chatty", "evidence_indices": [0]}
    ])
    mock = MockLLMClient(responses=[legacy])
    result = consolidate_feedback(feedback_list, "test.prompt.txt", mock)
    assert len(result.issues) == 1
    assert result.issues[0].category == "chatty-preamble"


def test_feedback_themes_round_trip():
    """FeedbackConfig.themes is emitted in to_yaml_dict iff non-empty, and parses back."""
    from prompterator.config.schema import Config, FeedbackConfig

    c_empty = Config()
    assert "themes" not in c_empty.to_yaml_dict()["feedback"]

    c_themed = Config(feedback=FeedbackConfig(themes=["tone", "structure"]))
    assert c_themed.to_yaml_dict()["feedback"]["themes"] == ["tone", "structure"]

    # Round-trip: re-build from the dict
    yaml_dict = c_themed.to_yaml_dict()
    rebuilt = Config(feedback=FeedbackConfig(**yaml_dict["feedback"]))
    assert rebuilt.feedback.themes == ["tone", "structure"]



def test_issue_evidence_round_trips_instance_and_confidence(tmp_path):
    """Save → load preserves anchor fields when present and omits them when absent."""
    from prompterator.core.issue import save_issue_file, load_issue_file
    from prompterator.models.issue import Issue, IssueEvidence, IssueFile

    issue_file = IssueFile(
        version="1.0",
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="tone",
                severity="medium",
                summary="rambling",
                evidence=[
                    IssueEvidence(
                        source="r1.mb",
                        feedback="rambling tone",
                        instance="rambling cadence",
                        confidence="high",
                    ),
                    IssueEvidence(source="r2.mb", feedback="too chatty"),
                ],
            ),
        ],
    )
    path = tmp_path / "test.issue.yaml"
    save_issue_file(issue_file, path)

    loaded = load_issue_file(path)
    ev0 = loaded.issues[0].evidence[0]
    ev1 = loaded.issues[0].evidence[1]
    assert ev0.instance == "rambling cadence"
    assert ev0.confidence == "high"
    assert ev1.instance is None
    assert ev1.confidence is None
