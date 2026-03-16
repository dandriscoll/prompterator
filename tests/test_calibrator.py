"""Tests for calibration logic."""

import tempfile
from pathlib import Path

import yaml

from prompterator.core.calibrator import (
    calibrate,
    classify_labels,
    compute_metrics,
    determine_verdict,
    run_calibration_eval,
    save_calibration_report,
)
from prompterator.models.calibration import (
    CalibrationExample,
    CalibrationReport,
    CalibrationResult,
)
from prompterator.models.eval import Eval, EvalFile, EvalRubric
from prompterator.models.feedback import Feedback, FeedbackEntry
from prompterator.models.issue import Issue, IssueEvidence, IssueFile

from tests.conftest import MockLLMClient


# ---------------------------------------------------------------------------
# Fixtures (inline, since they're calibration-specific)
# ---------------------------------------------------------------------------

def _make_feedback(name: str, text: str, prompt_ref: str = "test.prompt.txt") -> Feedback:
    return Feedback(
        source_file=name,
        prompt_ref=prompt_ref,
        entries=[FeedbackEntry(text=text)],
    )


def _make_issue_file(negative_sources: list[str]) -> IssueFile:
    """Build an IssueFile whose single issue lists *negative_sources* as evidence."""
    return IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="pose-misalignment",
                severity="high",
                summary="did not follow pose",
                evidence=[
                    IssueEvidence(source=src, feedback="did not follow pose")
                    for src in negative_sources
                ],
            ),
        ],
    )


def _make_eval(eval_id: str = "eval-test-01", issue_ref: str = "issue-test-01") -> Eval:
    return Eval(
        id=eval_id,
        type="rubric",
        issue_ref=issue_ref,
        rubric=EvalRubric(
            criteria=["Prompt addresses: did not follow pose"],
            scoring="all_required",
        ),
    )


# ---------------------------------------------------------------------------
# classify_labels
# ---------------------------------------------------------------------------

def test_classify_labels_only_evidence_sources():
    """Only files in issue evidence get labels; others are omitted."""
    feedback_list = [
        _make_feedback("001-r1.mb", "adds conversational preamble"),
        _make_feedback("001-r2.mb", "replaced checkboxes with sections"),
        _make_feedback("002-r1.mb", "output looks good"),
    ]
    issue_file = _make_issue_file(["001-r1.mb", "001-r2.mb"])
    eval_spec = _make_eval()

    labels = classify_labels(feedback_list, issue_file, eval_spec)
    assert labels["001-r1.mb"] == "FAIL"
    assert labels["001-r2.mb"] == "FAIL"
    assert "002-r1.mb" not in labels  # not assessed, not labeled


def test_classify_labels_no_issue_ref():
    """When eval has no issue_ref, no feedback is labelled."""
    feedback_list = [
        _make_feedback("file1.mb", "some feedback"),
        _make_feedback("file2.mb", "other feedback"),
    ]
    issue_file = _make_issue_file(["file1.mb"])
    eval_spec = Eval(
        id="eval-orphan",
        type="rubric",
        issue_ref=None,
        rubric=EvalRubric(criteria=["Something"], scoring="all_required"),
    )

    labels = classify_labels(feedback_list, issue_file, eval_spec)
    assert len(labels) == 0  # no labels without issue_ref


def test_classify_labels_full_path_normalisation():
    """Source files with full paths still match basename in evidence."""
    feedback_list = [
        _make_feedback("/data/feedback/001-r1.mb", "adds preamble"),
    ]
    issue_file = _make_issue_file(["001-r1.mb"])
    eval_spec = _make_eval()

    labels = classify_labels(feedback_list, issue_file, eval_spec)
    assert labels["001-r1.mb"] == "FAIL"


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect_detection():
    """All known-bad outputs detected gives detection rate 1.0."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="FAIL", match=True),
        CalibrationExample(source="b.mb", label="FAIL", eval_result="FAIL", match=True),
    ]
    detection, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert detection == 1.0
    assert recall == 1.0
    assert fp == 0
    assert fn == 0


def test_compute_metrics_with_misses():
    """Missed detections lower the detection rate."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="FAIL", match=True),   # TP
        CalibrationExample(source="b.mb", label="FAIL", eval_result="PASS", match=False),  # FN
    ]
    detection, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert detection == 0.5  # 1/2
    assert recall == 0.5
    assert fp == 0  # no PASS labels, so no FP possible
    assert fn == 1


def test_compute_metrics_all_missed():
    """All detections missed gives 0.0."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="PASS", match=False),
        CalibrationExample(source="b.mb", label="FAIL", eval_result="PASS", match=False),
    ]
    detection, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert detection == 0.0
    assert fn == 2
    assert fp == 0


def test_compute_metrics_empty():
    """Empty examples give zero detection rate."""
    detection, precision, recall, f1, fp, fn = compute_metrics([])
    assert detection == 0.0
    assert fp == 0
    assert fn == 0


# ---------------------------------------------------------------------------
# determine_verdict
# ---------------------------------------------------------------------------

def test_verdict_good():
    assert determine_verdict(0.80) == "GOOD"
    assert determine_verdict(0.95) == "GOOD"
    assert determine_verdict(1.0) == "GOOD"


def test_verdict_weak():
    assert determine_verdict(0.60) == "WEAK"
    assert determine_verdict(0.79) == "WEAK"


def test_verdict_bad():
    assert determine_verdict(0.59) == "BAD"
    assert determine_verdict(0.0) == "BAD"


# ---------------------------------------------------------------------------
# run_calibration_eval
# ---------------------------------------------------------------------------

def test_run_calibration_eval_pass():
    """LLM returning PASS should yield True."""
    llm = MockLLMClient(responses=[
        "CRITERION: Prompt addresses: did not follow pose\n"
        "RESULT: PASS\n"
        "REASON: Feedback says pose is correct\n"
        "OVERALL: PASS\nSCORE: 1.0"
    ])
    eval_spec = _make_eval()
    result = run_calibration_eval(eval_spec, "correct pose", llm)
    assert result is True


def test_run_calibration_eval_fail():
    """LLM returning FAIL should yield False."""
    llm = MockLLMClient(responses=[
        "CRITERION: Prompt addresses: did not follow pose\n"
        "RESULT: FAIL\n"
        "REASON: Feedback says pose was wrong\n"
        "OVERALL: FAIL\nSCORE: 0.0"
    ])
    eval_spec = _make_eval()
    result = run_calibration_eval(eval_spec, "did not follow pose", llm)
    assert result is False


def test_run_calibration_eval_non_rubric():
    """Non-rubric evals default to True."""
    llm = MockLLMClient()
    eval_spec = Eval(id="eval-assertion", type="assertion", assertion="something")
    result = run_calibration_eval(eval_spec, "some text", llm)
    assert result is True


# ---------------------------------------------------------------------------
# calibrate (end-to-end)
# ---------------------------------------------------------------------------

def test_calibrate_perfect_detection():
    """All known-bad feedback correctly detected."""
    feedback_list = [
        _make_feedback("neg1.mb", "did not follow pose"),
        _make_feedback("neg2.mb", "did not follow pose"),
        _make_feedback("pos1.mb", "correct pose"),  # not in evidence, skipped
    ]
    issue_file = _make_issue_file(["neg1.mb", "neg2.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # Only 2 LLM calls (neg1, neg2) — pos1 is skipped (no label)
    llm = MockLLMClient(responses=[
        "CRITERION: X\nRESULT: FAIL\nREASON: Bad\nOVERALL: FAIL\nSCORE: 0.0",
        "CRITERION: X\nRESULT: FAIL\nREASON: Bad\nOVERALL: FAIL\nSCORE: 0.0",
    ])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    assert len(results) == 1
    cal = results[0]
    assert cal.num_examples == 2
    assert cal.accuracy == 1.0
    assert cal.verdict == "GOOD"
    assert cal.false_positives == 0
    assert cal.false_negatives == 0


def test_calibrate_with_false_negative():
    """One negative example gets PASS from eval (missed detection)."""
    feedback_list = [
        _make_feedback("neg1.mb", "did not follow pose"),
        _make_feedback("neg2.mb", "did not follow pose"),
        _make_feedback("pos1.mb", "correct pose"),  # skipped
        _make_feedback("pos2.mb", "correct pose"),  # skipped
    ]
    issue_file = _make_issue_file(["neg1.mb", "neg2.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # Only 2 LLM calls: neg1=FAIL(correct), neg2=PASS(missed)
    llm = MockLLMClient(responses=[
        "CRITERION: X\nRESULT: FAIL\nREASON: Bad\nOVERALL: FAIL\nSCORE: 0.0",
        "CRITERION: X\nRESULT: PASS\nREASON: OK\nOVERALL: PASS\nSCORE: 1.0",
    ])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.num_examples == 2  # only the 2 known-FAIL sources
    assert cal.accuracy == 0.5  # 1/2 detected
    assert cal.false_negatives == 1
    assert cal.false_positives == 0
    assert cal.verdict == "BAD"


def test_calibrate_all_missed():
    """All known-bad examples missed gives BAD verdict."""
    feedback_list = [
        _make_feedback("neg1.mb", "bad"),
        _make_feedback("neg2.mb", "bad"),
        _make_feedback("pos1.mb", "good"),  # skipped
    ]
    issue_file = _make_issue_file(["neg1.mb", "neg2.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # Both return PASS — both missed
    llm = MockLLMClient(responses=[
        "CRITERION: X\nRESULT: PASS\nREASON: x\nOVERALL: PASS\nSCORE: 1.0",
        "CRITERION: X\nRESULT: PASS\nREASON: x\nOVERALL: PASS\nSCORE: 1.0",
    ])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.accuracy == 0.0
    assert cal.verdict == "BAD"
    assert cal.false_negatives == 2


def test_calibrate_skips_unassessed_feedback():
    """Feedback not in issue evidence is skipped, not labeled PASS."""
    feedback_list = [
        _make_feedback("neg1.mb", "did not follow pose"),
        _make_feedback("unrelated.mb", "different problem entirely"),
    ]
    issue_file = _make_issue_file(["neg1.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # Only 1 LLM call — unrelated.mb is skipped
    llm = MockLLMClient(responses=[
        "CRITERION: X\nRESULT: FAIL\nREASON: x\nOVERALL: FAIL\nSCORE: 0.0",
    ])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.num_examples == 1  # only neg1
    assert cal.accuracy == 1.0
    assert len(llm.calls) == 1  # confirm only 1 LLM call


def test_calibrate_no_evidence_produces_no_results():
    """Eval with no issue evidence produces no calibration result."""
    feedback_list = [
        _make_feedback("file1.mb", "some feedback"),
    ]
    issue_file = _make_issue_file([])  # no evidence
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    llm = MockLLMClient()
    results = calibrate(eval_file, feedback_list, issue_file, llm)
    assert len(results) == 0  # nothing to calibrate
    assert len(llm.calls) == 0  # no LLM calls made


def test_calibrate_multiple_evals():
    """Calibration runs independently per eval, each using only its evidence."""
    feedback_list = [
        _make_feedback("f1.mb", "unclear instructions"),
        _make_feedback("f2.mb", "missing examples"),
        _make_feedback("f3.mb", "good job"),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="clarity",
                severity="high",
                summary="unclear",
                evidence=[IssueEvidence(source="f1.mb", feedback="unclear instructions")],
            ),
            Issue(
                id="issue-test-02",
                category="completeness",
                severity="low",
                summary="missing examples",
                evidence=[IssueEvidence(source="f2.mb", feedback="missing examples")],
            ),
        ],
    )
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[
            _make_eval("eval-clarity-01", "issue-test-01"),
            _make_eval("eval-completeness-02", "issue-test-02"),
        ],
    )

    # 2 LLM calls: 1 per eval (each eval has 1 evidence source)
    llm = MockLLMClient(responses=[
        # eval-clarity-01: f1=FAIL(correct)
        "CRITERION: X\nRESULT: FAIL\nREASON: x\nOVERALL: FAIL\nSCORE: 0.0",
        # eval-completeness-02: f2=FAIL(correct)
        "CRITERION: X\nRESULT: FAIL\nREASON: x\nOVERALL: FAIL\nSCORE: 0.0",
    ])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    assert len(results) == 2
    assert results[0].eval_id == "eval-clarity-01"
    assert results[0].accuracy == 1.0
    assert results[0].num_examples == 1
    assert results[1].eval_id == "eval-completeness-02"
    assert results[1].accuracy == 1.0
    assert results[1].num_examples == 1
    assert len(llm.calls) == 2  # one call per evidence source


# ---------------------------------------------------------------------------
# YAML output serialization
# ---------------------------------------------------------------------------

def test_calibration_report_yaml_serialization():
    """CalibrationReport serializes to valid YAML."""
    report = CalibrationReport(
        prompt_ref="test.prompt.txt",
        eval_file="evals/test.eval.yaml",
        calibrations=[
            CalibrationResult(
                eval_id="eval-01",
                num_examples=2,
                accuracy=0.5,
                precision=1.0,
                recall=0.5,
                f1=0.6667,
                false_positives=0,
                false_negatives=1,
                verdict="BAD",
                examples=[
                    CalibrationExample(source="a.mb", label="FAIL", eval_result="FAIL", match=True),
                    CalibrationExample(source="b.mb", label="FAIL", eval_result="PASS", match=False),
                ],
            ),
        ],
    )

    d = report.to_yaml_dict()
    assert d["version"] == "1.0"
    assert d["prompt_ref"] == "test.prompt.txt"
    assert len(d["calibrations"]) == 1
    cal = d["calibrations"][0]
    assert cal["eval_id"] == "eval-01"
    assert cal["accuracy"] == 0.5
    assert len(cal["examples"]) == 2
    assert cal["examples"][1]["match"] is False


def test_save_calibration_report(tmp_path):
    """Report saves to disk as valid YAML."""
    report = CalibrationReport(
        prompt_ref="test.prompt.txt",
        eval_file="evals/test.eval.yaml",
        calibrations=[
            CalibrationResult(
                eval_id="eval-01",
                num_examples=2,
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                false_positives=0,
                false_negatives=0,
                verdict="GOOD",
                examples=[
                    CalibrationExample(source="a.mb", label="FAIL", eval_result="FAIL", match=True),
                    CalibrationExample(source="b.mb", label="FAIL", eval_result="FAIL", match=True),
                ],
            ),
        ],
    )

    out = tmp_path / "results" / "test.calibration.yaml"
    save_calibration_report(report, out)

    assert out.exists()
    with open(out) as f:
        data = yaml.safe_load(f)
    assert data["version"] == "1.0"
    assert data["calibrations"][0]["verdict"] == "GOOD"
    assert len(data["calibrations"][0]["examples"]) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_detection_rate_zero_when_all_missed():
    """Detection rate is 0 when eval misses every known-bad example."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="PASS", match=False),
        CalibrationExample(source="b.mb", label="FAIL", eval_result="PASS", match=False),
    ]
    detection, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert detection == 0.0
    assert recall == 0.0
    assert f1 == 0.0
    assert fp == 0
    assert fn == 2
