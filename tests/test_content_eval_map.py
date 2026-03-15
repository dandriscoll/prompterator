"""Tests for feedback-based content-to-eval mapping and selective eval running."""

import tempfile
from pathlib import Path

from prompterator.core.eval_runner import (
    map_content_to_evals,
    run_all_evals,
)
from prompterator.models.eval import Eval, EvalFile, EvalRubric
from prompterator.models.feedback import Feedback, FeedbackEntry
from prompterator.models.issue import Issue, IssueEvidence, IssueFile

from tests.conftest import MockLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_content(name: str, tmp_dir: Path) -> Path:
    """Create a content file and return its path."""
    p = tmp_dir / name
    p.write_text(f"Content of {name}")
    return p


def _make_feedback(mb_name: str, prior_ref: str, text: str) -> Feedback:
    return Feedback(
        source_file=mb_name,
        prompt_ref="test.prompt.txt",
        entries=[FeedbackEntry(text=text, prior_ref=prior_ref)],
    )


def _make_issue_file(issues: list[tuple[str, list[str]]]) -> IssueFile:
    """Build IssueFile from list of (issue_id, [evidence_sources])."""
    return IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id=issue_id,
                category="test",
                severity="high",
                summary=f"Issue {issue_id}",
                evidence=[
                    IssueEvidence(source=src, feedback="test feedback")
                    for src in sources
                ],
            )
            for issue_id, sources in issues
        ],
    )


def _make_eval_file(evals: list[tuple[str, str]]) -> EvalFile:
    """Build EvalFile from list of (eval_id, issue_ref)."""
    return EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[
            Eval(
                id=eval_id,
                type="rubric",
                issue_ref=issue_ref,
                rubric=EvalRubric(criteria=[f"Check for {eval_id}"]),
            )
            for eval_id, issue_ref in evals
        ],
    )


# ---------------------------------------------------------------------------
# map_content_to_evals
# ---------------------------------------------------------------------------

def test_map_content_all_covered(tmp_path):
    """All content has feedback — returns mapping."""
    content_paths = [
        _make_content("doc01.md", tmp_path),
        _make_content("doc02.md", tmp_path),
    ]
    feedback_list = [
        _make_feedback("review01.mb", "doc01.md", "vague wording"),
        _make_feedback("review02.mb", "doc02.md", "missing examples"),
    ]
    issue_file = _make_issue_file([
        ("issue-01", ["review01.mb"]),
        ("issue-02", ["review02.mb"]),
    ])
    eval_file = _make_eval_file([
        ("eval-01", "issue-01"),
        ("eval-02", "issue-02"),
    ])

    result = map_content_to_evals(content_paths, feedback_list, issue_file, eval_file)
    assert result is not None
    assert result[0] == ["eval-01"]  # doc01 → review01 → issue-01 → eval-01
    assert result[1] == ["eval-02"]  # doc02 → review02 → issue-02 → eval-02


def test_map_content_missing_feedback(tmp_path):
    """Content without feedback — returns None (run all evals)."""
    content_paths = [
        _make_content("doc01.md", tmp_path),
        _make_content("doc02.md", tmp_path),
    ]
    # Only doc01 has feedback
    feedback_list = [
        _make_feedback("review01.mb", "doc01.md", "vague wording"),
    ]
    issue_file = _make_issue_file([("issue-01", ["review01.mb"])])
    eval_file = _make_eval_file([("eval-01", "issue-01")])

    result = map_content_to_evals(content_paths, feedback_list, issue_file, eval_file)
    assert result is None


def test_map_content_shared_feedback(tmp_path):
    """One feedback file covers both content and issues."""
    content_paths = [_make_content("doc01.md", tmp_path)]
    feedback_list = [
        _make_feedback("review01.mb", "doc01.md", "vague wording"),
    ]
    # review01 is evidence for both issues
    issue_file = _make_issue_file([
        ("issue-01", ["review01.mb"]),
        ("issue-02", ["review01.mb"]),
    ])
    eval_file = _make_eval_file([
        ("eval-01", "issue-01"),
        ("eval-02", "issue-02"),
        ("eval-03", "issue-03"),  # issue-03 not linked to review01
    ])

    result = map_content_to_evals(content_paths, feedback_list, issue_file, eval_file)
    assert result is not None
    # doc01 → review01 → issue-01, issue-02 → eval-01, eval-02
    assert set(result[0]) == {"eval-01", "eval-02"}


def test_map_content_no_feedback(tmp_path):
    """No feedback at all — returns None."""
    content_paths = [_make_content("doc01.md", tmp_path)]
    result = map_content_to_evals(content_paths, [], IssueFile(prompt_ref="x", issues=[]),
                                   _make_eval_file([("eval-01", "issue-01")]))
    assert result is None


def test_map_content_multiple_feedback_per_content(tmp_path):
    """Content has feedback from multiple .mb files."""
    content_paths = [_make_content("doc01.md", tmp_path)]
    feedback_list = [
        _make_feedback("review01.mb", "doc01.md", "vague"),
        _make_feedback("review02.mb", "doc01.md", "missing examples"),
    ]
    issue_file = _make_issue_file([
        ("issue-01", ["review01.mb"]),
        ("issue-02", ["review02.mb"]),
    ])
    eval_file = _make_eval_file([
        ("eval-01", "issue-01"),
        ("eval-02", "issue-02"),
    ])

    result = map_content_to_evals(content_paths, feedback_list, issue_file, eval_file)
    assert result is not None
    assert set(result[0]) == {"eval-01", "eval-02"}


def test_map_content_empty_eval_set(tmp_path):
    """Content has feedback but no evals trace through."""
    content_paths = [_make_content("doc01.md", tmp_path)]
    feedback_list = [
        _make_feedback("review01.mb", "doc01.md", "ok"),
    ]
    # Issue references review01, but no eval references issue-01
    issue_file = _make_issue_file([("issue-01", ["review01.mb"])])
    eval_file = _make_eval_file([("eval-99", "issue-99")])  # different issue

    result = map_content_to_evals(content_paths, feedback_list, issue_file, eval_file)
    assert result is not None
    assert result[0] == []  # No evals apply


def test_map_content_full_path_prior_ref(tmp_path):
    """prior_ref with full path still matches content basename."""
    content_paths = [_make_content("doc01.md", tmp_path)]
    feedback_list = [
        Feedback(
            source_file="review01.mb",
            prompt_ref="test.prompt.txt",
            entries=[FeedbackEntry(
                text="feedback",
                prior_ref="/some/full/path/doc01.md",
            )],
        ),
    ]
    issue_file = _make_issue_file([("issue-01", ["review01.mb"])])
    eval_file = _make_eval_file([("eval-01", "issue-01")])

    result = map_content_to_evals(content_paths, feedback_list, issue_file, eval_file)
    assert result is not None
    assert result[0] == ["eval-01"]


# ---------------------------------------------------------------------------
# run_all_evals with content_eval_map
# ---------------------------------------------------------------------------

def test_run_all_evals_with_filtering():
    """Only mapped evals run for each content; unmapped evals are skipped."""
    eval_file = _make_eval_file([
        ("eval-01", "issue-01"),
        ("eval-02", "issue-02"),
        ("eval-03", "issue-03"),
    ])

    # 2 content files: content 0 runs eval-01 only, content 1 runs eval-02 only
    # eval-03 is not mapped to any content → skipped
    content_eval_map = {
        0: ["eval-01"],
        1: ["eval-02"],
    }

    # Mock LLM: all pass
    llm = MockLLMClient(responses=[
        # eval-01 for content 0
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
        # eval-02 for content 1
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test prompt")
        tmp = Path(f.name)

    try:
        result_file = run_all_evals(
            eval_file, tmp, llm,
            content_texts=["content one", "content two"],
            ensemble=1,
            content_eval_map=content_eval_map,
        )
    finally:
        tmp.unlink()

    # Only eval-01 and eval-02 should be in results (eval-03 skipped)
    result_ids = {r.eval_id for r in result_file.results}
    assert result_ids == {"eval-01", "eval-02"}

    # Both should pass
    for r in result_file.results:
        assert r.passed


def test_run_all_evals_no_filtering():
    """Without content_eval_map, all evals run for all content."""
    eval_file = _make_eval_file([
        ("eval-01", "issue-01"),
        ("eval-02", "issue-02"),
    ])

    # 3 LLM calls needed: 1 content * 2 evals * 1 ensemble
    llm = MockLLMClient(responses=[
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test prompt")
        tmp = Path(f.name)

    try:
        result_file = run_all_evals(
            eval_file, tmp, llm,
            content_texts=["content one"],
            ensemble=1,
        )
    finally:
        tmp.unlink()

    # Both evals should be in results
    result_ids = {r.eval_id for r in result_file.results}
    assert result_ids == {"eval-01", "eval-02"}


def test_run_all_evals_empty_mapping_for_content():
    """Content with empty eval list produces no eval results for that content."""
    eval_file = _make_eval_file([
        ("eval-01", "issue-01"),
    ])

    content_eval_map = {
        0: [],      # no evals for content 0
        1: ["eval-01"],  # eval-01 for content 1
    }

    llm = MockLLMClient(responses=[
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test prompt")
        tmp = Path(f.name)

    try:
        result_file = run_all_evals(
            eval_file, tmp, llm,
            content_texts=["content one", "content two"],
            ensemble=1,
            content_eval_map=content_eval_map,
        )
    finally:
        tmp.unlink()

    # eval-01 runs for content 1 only → 1 data point
    assert len(result_file.results) == 1
    assert result_file.results[0].eval_id == "eval-01"
    assert result_file.results[0].passed


# ---------------------------------------------------------------------------
# resolve_content_with_paths
# ---------------------------------------------------------------------------

def test_resolve_content_with_paths_returns_paths_and_texts(tmp_path):
    """resolve_content_with_paths returns (path, text) tuples."""
    from prompterator.commands.resolve import resolve_content_with_paths
    from prompterator.config.schema import Config

    doc = tmp_path / "doc.md"
    doc.write_text("hello world")

    config = Config(directories={"content": [str(doc)]})
    result = resolve_content_with_paths(config, tmp_path)

    assert len(result) == 1
    assert result[0][0] == doc
    assert result[0][1] == "hello world"


def test_resolve_content_with_paths_cli_override(tmp_path):
    """CLI content flag overrides config."""
    from prompterator.commands.resolve import resolve_content_with_paths
    from prompterator.config.schema import Config

    doc = tmp_path / "cli_doc.md"
    doc.write_text("cli content")

    config = Config()
    result = resolve_content_with_paths(config, tmp_path, cli_content=doc)

    assert len(result) == 1
    assert result[0][0] == doc
    assert result[0][1] == "cli content"


def test_resolve_content_with_paths_empty(tmp_path):
    """No content configured returns empty list."""
    from prompterator.commands.resolve import resolve_content_with_paths
    from prompterator.config.schema import Config

    config = Config()
    result = resolve_content_with_paths(config, tmp_path)
    assert result == []
