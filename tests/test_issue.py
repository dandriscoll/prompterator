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
from prompterator.models.issue import Issue, IssueEvidence


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


# ---------------------------------------------------------------------------
# Polarity: positive observations cluster to the same axis as negatives
# ---------------------------------------------------------------------------

def test_cluster_prompt_documents_polarity():
    """Tripwire: clustering prompt must describe both polarities."""
    assert "polarity" in _CLUSTER_SYSTEM
    assert "negative" in _CLUSTER_SYSTEM and "positive" in _CLUSTER_SYSTEM
    # Must no longer instruct the LLM to skip positives.
    assert "Skip positive observations" not in _CLUSTER_SYSTEM


def test_positive_anchors_become_positive_evidence():
    """An anchor tagged polarity=positive lands in IssueEvidence with polarity=positive."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="clothes distorted (appear ripped)")],
        ),
        Feedback(
            source_file="r2.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="no clothing distortions")],
        ),
    ]
    response = _themed_response(
        anchors=[
            {"index": 0, "instance": "clothes ripped", "polarity": "negative",
             "confidence": "high", "themes": ["clothing"]},
            {"index": 1, "instance": "no clothing distortions", "polarity": "positive",
             "confidence": "high", "themes": ["clothing"]},
        ],
        clusters=[
            {"theme": "clothing", "failure_mode": "distortion",
             "summary": "clothing distortion axis", "anchor_indices": [0, 1]}
        ],
    )
    mock = MockLLMClient(responses=[response])

    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, themes=["clothing"]
    )
    assert len(result.issues) == 1
    polarities = sorted(ev.polarity for ev in result.issues[0].evidence)
    assert polarities == ["negative", "positive"]


def test_pure_positive_cluster_is_dropped():
    """A cluster whose anchors are all positive has no problem to fix → no issue."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="face generated correctly")],
        ),
    ]
    response = _themed_response(
        anchors=[
            {"index": 0, "instance": "face correct", "polarity": "positive",
             "confidence": "high", "themes": ["face-quality"]},
        ],
        clusters=[
            {"theme": "face-quality", "failure_mode": "n/a",
             "summary": "face quality affirmed", "anchor_indices": [0]}
        ],
    )
    mock = MockLLMClient(responses=[response])
    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, themes=["face-quality"]
    )
    assert result.issues == []


def test_severity_counts_only_negative_sources():
    """Severity uses negative-evidence sources, not total evidence count."""
    # 2 sources: 1 has a negative observation, 1 has a positive one.
    # Severity should reflect 1/2 = medium, not 2/2 = high.
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="clothes distorted")],
        ),
        Feedback(
            source_file="r2.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="no clothing distortions")],
        ),
    ]
    response = _themed_response(
        anchors=[
            {"index": 0, "instance": "clothes distorted", "polarity": "negative",
             "confidence": "high", "themes": ["clothing"]},
            {"index": 1, "instance": "clean", "polarity": "positive",
             "confidence": "high", "themes": ["clothing"]},
        ],
        clusters=[
            {"theme": "clothing", "failure_mode": "distortion",
             "summary": "distortion axis", "anchor_indices": [0, 1]}
        ],
    )
    mock = MockLLMClient(responses=[response])
    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, themes=["clothing"]
    )
    assert result.issues[0].severity == "medium"  # 1 negative / 2 total → medium


def test_missing_polarity_defaults_to_negative():
    """Anchors without polarity are treated as negative (legacy-response backstop)."""
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="clothes distorted")],
        ),
    ]
    response = _themed_response(
        anchors=[
            # No polarity field.
            {"index": 0, "instance": "clothes distorted", "confidence": "high",
             "themes": ["clothing"]},
        ],
        clusters=[
            {"theme": "clothing", "failure_mode": "distortion",
             "summary": "distortion axis", "anchor_indices": [0]}
        ],
    )
    mock = MockLLMClient(responses=[response])
    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock, themes=["clothing"]
    )
    assert result.issues[0].evidence[0].polarity == "negative"


# ---------------------------------------------------------------------------
# Over-counting: repeat runs with existing_issues must not duplicate evidence
# ---------------------------------------------------------------------------

def test_duplicate_fragments_in_new_feedback_kept():
    """Two entries in the same .mb file with the same fragment produce two observations."""
    # Entries 2 and 5 both contain "prompt not followed" — severity accounting
    # needs both to surface as separate observations so the LLM can count
    # how often the problem happened.
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(text="prompt not followed"),
                FeedbackEntry(text="prompt not followed"),
            ],
        ),
    ]
    captured: list[str] = []

    class _CaptureLLM(MockLLMClient):
        def generate(self, prompt, **kwargs):
            captured.append(prompt)
            return super().generate(prompt, **kwargs)

    response = _themed_response(
        anchors=[
            {"index": 0, "instance": "prompt not followed", "polarity": "negative",
             "confidence": "high", "themes": ["prompt"]},
            {"index": 1, "instance": "prompt not followed", "polarity": "negative",
             "confidence": "high", "themes": ["prompt"]},
        ],
        clusters=[
            {"theme": "prompt", "failure_mode": "not-followed",
             "summary": "prompt not followed", "anchor_indices": [0, 1]}
        ],
    )
    mock = _CaptureLLM(responses=[response])
    consolidate_feedback(feedback_list, "test.prompt.txt", mock, themes=["prompt"])

    obs_lines = [
        line for line in captured[0].splitlines()
        if "prompt not followed" in line and line.startswith("[")
    ]
    assert len(obs_lines) == 2, obs_lines


def test_existing_and_new_observations_deduped():
    """When existing issues' evidence overlaps new feedback, observations aren't doubled."""
    # Simulate a second `prompterator issues` run: existing evidence lists the
    # same entry text that the fresh feedback also contains.
    existing = [
        Issue(
            id="issue-test-01",
            category="clothing",
            severity="high",
            summary="clothing distorted",
            evidence=[
                IssueEvidence(
                    source="r1.mb", feedback="clothes distorted", polarity="negative"
                ),
            ],
        ),
    ]
    feedback_list = [
        Feedback(
            source_file="r1.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(text="clothes distorted")],
        ),
    ]
    # Capture the prompt the LLM sees to verify the observation appears only once.
    captured_prompts: list[str] = []

    class _CaptureLLM(MockLLMClient):
        def generate(self, prompt, **kwargs):
            captured_prompts.append(prompt)
            return super().generate(prompt, **kwargs)

    response = _themed_response(
        anchors=[
            {"index": 0, "instance": "clothes distorted", "polarity": "negative",
             "confidence": "high", "themes": ["clothing"]},
        ],
        clusters=[
            {"theme": "clothing", "failure_mode": "distortion",
             "summary": "distortion axis", "anchor_indices": [0]}
        ],
    )
    mock = _CaptureLLM(responses=[response])
    result = consolidate_feedback(
        feedback_list, "test.prompt.txt", mock,
        existing_issues=existing, themes=["clothing"],
    )

    # Exactly one observation for (r1.mb, "clothes distorted") — not two.
    assert captured_prompts
    obs_lines = [
        line for line in captured_prompts[0].splitlines()
        if "clothes distorted" in line and line.startswith("[")
    ]
    assert len(obs_lines) == 1, obs_lines
    # And the resulting issue has a single evidence record, not two.
    assert len(result.issues) == 1
    assert len(result.issues[0].evidence) == 1


def test_issue_evidence_round_trips_polarity(tmp_path):
    """Polarity survives save + reload; default is negative when absent from disk."""
    from prompterator.core.issue import save_issue_file, load_issue_file
    from prompterator.models.issue import Issue, IssueEvidence, IssueFile

    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="clothing",
                severity="high",
                summary="distortion",
                evidence=[
                    IssueEvidence(
                        source="r1.mb", feedback="clothes distorted", polarity="negative"
                    ),
                    IssueEvidence(
                        source="r2.mb", feedback="no distortions", polarity="positive"
                    ),
                ],
            ),
        ],
    )
    path = tmp_path / "test.issue.yaml"
    save_issue_file(issue_file, path)

    # Hand-written yaml without polarity still loads, defaults to negative.
    legacy = tmp_path / "legacy.issue.yaml"
    legacy.write_text(
        "version: '1.0'\n"
        "prompt_ref: test.prompt.txt\n"
        "issues:\n"
        "- id: issue-test-01\n"
        "  category: clothing\n"
        "  severity: high\n"
        "  summary: legacy\n"
        "  evidence:\n"
        "  - source: r1.mb\n"
        "    feedback: old format\n"
    )

    loaded = load_issue_file(path)
    assert loaded.issues[0].evidence[0].polarity == "negative"
    assert loaded.issues[0].evidence[1].polarity == "positive"

    legacy_loaded = load_issue_file(legacy)
    assert legacy_loaded.issues[0].evidence[0].polarity == "negative"


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
