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
                category="chatty-preamble",
                severity="high",
                summary="adds conversational preamble",
                evidence=[
                    IssueEvidence(source=src, feedback="adds conversational preamble")
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
            criteria=["Output does not begin with a conversational preamble"],
            scoring="all_required",
        ),
    )


# ---------------------------------------------------------------------------
# classify_labels
# ---------------------------------------------------------------------------

def test_classify_labels_only_evidence_entries():
    """Only specific entries cited as evidence are labeled; others omitted."""
    feedback_list = [
        _make_feedback("001-r1.mb", "adds conversational preamble"),
        _make_feedback("001-r2.mb", "grammar issues"),  # different category
        _make_feedback("002-r1.mb", "output looks good"),
    ]
    # Issue evidence cites 001-r1.mb with specific text
    issue_file = _make_issue_file(["001-r1.mb"])
    eval_spec = _make_eval()

    evidence = classify_labels(feedback_list, issue_file, eval_spec)
    assert ("001-r1.mb", "adds conversational preamble") in evidence
    # 001-r2.mb has different text, not in evidence
    assert ("001-r2.mb", "grammar issues") not in evidence
    assert ("002-r1.mb", "output looks good") not in evidence


def test_classify_labels_no_issue_ref():
    """When eval has no issue_ref, no entries are labeled."""
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

    evidence = classify_labels(feedback_list, issue_file, eval_spec)
    assert len(evidence) == 0


def test_classify_labels_full_path_normalisation():
    """Source files with full paths still match basename in evidence."""
    feedback_list = [
        _make_feedback("/data/feedback/001-r1.mb", "adds conversational preamble"),
    ]
    issue_file = _make_issue_file(["001-r1.mb"])
    eval_spec = _make_eval()

    evidence = classify_labels(feedback_list, issue_file, eval_spec)
    assert ("001-r1.mb", "adds conversational preamble") in evidence


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect():
    """All TP and TN gives accuracy, precision, recall all 1.0."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="FAIL", match=True),   # TP
        CalibrationExample(source="b.mb", label="FAIL", eval_result="FAIL", match=True),   # TP
        CalibrationExample(source="c.mb", label="PASS", eval_result="PASS", match=True),   # TN
    ]
    accuracy, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert accuracy == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert fp == 0
    assert fn == 0


def test_compute_metrics_with_misses():
    """Missed detections lower recall but not precision."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="FAIL", match=True),   # TP
        CalibrationExample(source="b.mb", label="FAIL", eval_result="PASS", match=False),  # FN
    ]
    accuracy, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert accuracy == 0.5  # (1+0) / 2
    assert precision == 1.0  # no FP predictions
    assert recall == 0.5  # 1/2 FAIL-labeled caught
    assert fp == 0
    assert fn == 1


def test_compute_metrics_with_false_positives():
    """False alarms on clean outputs lower precision but not recall."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="FAIL", match=True),   # TP
        CalibrationExample(source="b.mb", label="PASS", eval_result="FAIL", match=False),  # FP
        CalibrationExample(source="c.mb", label="PASS", eval_result="PASS", match=True),   # TN
    ]
    accuracy, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert accuracy == 2 / 3
    assert precision == 0.5  # 1 TP / (1 TP + 1 FP)
    assert recall == 1.0  # no FAIL missed
    assert fp == 1
    assert fn == 0


def test_compute_metrics_all_missed():
    """All FAIL labels missed gives recall 0."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="PASS", match=False),  # FN
        CalibrationExample(source="b.mb", label="FAIL", eval_result="PASS", match=False),  # FN
    ]
    accuracy, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert accuracy == 0.0
    assert recall == 0.0
    assert precision == 1.0  # no FP predictions
    assert fn == 2
    assert fp == 0


def test_compute_metrics_empty():
    """Empty examples give zero accuracy but default precision/recall to 1.0."""
    accuracy, precision, recall, f1, fp, fn = compute_metrics([])
    assert accuracy == 0.0
    assert precision == 1.0
    assert recall == 1.0
    assert fp == 0
    assert fn == 0


# ---------------------------------------------------------------------------
# determine_verdict
# ---------------------------------------------------------------------------

def test_verdict_good():
    assert determine_verdict(1.0, 1.0) == "GOOD"
    assert determine_verdict(0.80, 0.80) == "GOOD"
    assert determine_verdict(0.95, 0.85) == "GOOD"


def test_verdict_weak():
    assert determine_verdict(0.79, 0.80) == "WEAK"  # one side weak
    assert determine_verdict(0.60, 0.60) == "WEAK"
    assert determine_verdict(1.0, 0.70) == "WEAK"   # recall the bottleneck


def test_verdict_bad():
    assert determine_verdict(0.59, 1.0) == "BAD"    # precision the bottleneck
    assert determine_verdict(1.0, 0.0) == "BAD"
    assert determine_verdict(0.5, 0.5) == "BAD"


# ---------------------------------------------------------------------------
# run_calibration_eval
# ---------------------------------------------------------------------------

def test_run_calibration_eval_pass():
    """LLM returning PASS should yield True."""
    llm = MockLLMClient(responses=[
        "CRITERION: Output does not begin with a conversational preamble\n"
        "RESULT: PASS\n"
        "REASON: No preamble found\n"
        "OVERALL: PASS\nSCORE: 1.0"
    ])
    eval_spec = _make_eval()
    result = run_calibration_eval(eval_spec, "output starts directly with the list", llm)
    assert result is True


def test_run_calibration_eval_fail():
    """LLM returning FAIL should yield False."""
    llm = MockLLMClient(responses=[
        "CRITERION: Output does not begin with a conversational preamble\n"
        "RESULT: FAIL\n"
        "REASON: Output starts with chatty intro\n"
        "OVERALL: FAIL\nSCORE: 0.0"
    ])
    eval_spec = _make_eval()
    result = run_calibration_eval(eval_spec, "adds conversational preamble", llm)
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

_FAIL_RESP = "CRITERION: X\nRESULT: FAIL\nREASON: Bad\nOVERALL: FAIL\nSCORE: 0.0"
_PASS_RESP = "CRITERION: X\nRESULT: PASS\nREASON: OK\nOVERALL: PASS\nSCORE: 1.0"


def test_calibrate_perfect_detection():
    """All FAIL entries caught, all PASS entries cleared."""
    feedback_list = [
        _make_feedback("neg1.mb", "adds conversational preamble"),
        _make_feedback("neg2.mb", "adds conversational preamble"),
        _make_feedback("pos1.mb", "output starts directly with the list"),
    ]
    issue_file = _make_issue_file(["neg1.mb", "neg2.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # neg1=FAIL(TP), neg2=FAIL(TP), pos1=PASS(TN)
    llm = MockLLMClient(responses=[_FAIL_RESP, _FAIL_RESP, _PASS_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    assert len(results) == 1
    cal = results[0]
    assert cal.num_examples == 3
    assert cal.accuracy == 1.0
    assert cal.precision == 1.0
    assert cal.recall == 1.0
    assert cal.verdict == "GOOD"
    assert cal.false_positives == 0
    assert cal.false_negatives == 0


def test_calibrate_with_false_negative():
    """One FAIL entry missed by the eval; PASS entries still cleared."""
    feedback_list = [
        _make_feedback("neg1.mb", "adds conversational preamble"),
        _make_feedback("neg2.mb", "adds conversational preamble"),
        _make_feedback("pos1.mb", "output starts directly with the list"),
        _make_feedback("pos2.mb", "output starts directly with the list"),
    ]
    issue_file = _make_issue_file(["neg1.mb", "neg2.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # neg1=FAIL(TP), neg2=PASS(FN), pos1=PASS(TN), pos2=PASS(TN)
    llm = MockLLMClient(responses=[_FAIL_RESP, _PASS_RESP, _PASS_RESP, _PASS_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.num_examples == 4
    assert cal.accuracy == 0.75  # 3/4 correct
    assert cal.recall == 0.5  # 1/2 FAIL caught
    assert cal.precision == 1.0  # no FP
    assert cal.false_negatives == 1
    assert cal.false_positives == 0
    assert cal.verdict == "BAD"  # recall drives verdict down


def test_calibrate_with_false_positive():
    """Eval fires on a clean output — false alarm lowers precision."""
    feedback_list = [
        _make_feedback("neg1.mb", "adds conversational preamble"),
        _make_feedback("pos1.mb", "output starts directly with the list"),
    ]
    issue_file = _make_issue_file(["neg1.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # neg1=FAIL(TP), pos1=FAIL(FP — false alarm on clean output)
    llm = MockLLMClient(responses=[_FAIL_RESP, _FAIL_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.num_examples == 2
    assert cal.precision == 0.5  # 1 TP / (1 TP + 1 FP)
    assert cal.recall == 1.0
    assert cal.false_positives == 1
    assert cal.false_negatives == 0
    assert cal.verdict == "BAD"


def test_calibrate_all_missed():
    """All known-bad examples missed gives BAD verdict."""
    feedback_list = [
        _make_feedback("neg1.mb", "adds conversational preamble"),
        _make_feedback("neg2.mb", "adds conversational preamble"),
        _make_feedback("pos1.mb", "good"),
    ]
    issue_file = _make_issue_file(["neg1.mb", "neg2.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # neg1=PASS(FN), neg2=PASS(FN), pos1=PASS(TN)
    llm = MockLLMClient(responses=[_PASS_RESP, _PASS_RESP, _PASS_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.recall == 0.0
    assert cal.verdict == "BAD"
    assert cal.false_negatives == 2


def test_calibrate_ignores_positive_evidence_for_fail_matching():
    """Positive-polarity evidence records do not cause entries to be labeled FAIL."""
    # Evidence for the issue: one negative ("bad"), one positive ("clean").
    # Only the negative entry should be labeled FAIL.
    feedback_list = [
        _make_feedback("bad.mb", "adds conversational preamble"),
        _make_feedback("clean.mb", "output starts directly with the list"),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="preamble",
                severity="high",
                summary="preamble axis",
                evidence=[
                    IssueEvidence(
                        source="bad.mb",
                        feedback="adds conversational preamble",
                        polarity="negative",
                    ),
                    IssueEvidence(
                        source="clean.mb",
                        feedback="output starts directly with the list",
                        polarity="positive",
                    ),
                ],
            ),
        ],
    )
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # bad=FAIL(TP), clean=PASS(TN)
    llm = MockLLMClient(responses=[_FAIL_RESP, _PASS_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    labels = {ex.source: ex.label for ex in cal.examples}
    assert labels == {"bad.mb": "FAIL", "clean.mb": "PASS"}
    assert cal.accuracy == 1.0


def test_calibrate_skips_eval_with_only_positive_evidence():
    """An eval whose linked issue has only positive evidence produces no calibration."""
    feedback_list = [
        _make_feedback("f1.mb", "output starts directly with the list"),
    ]
    issue_file = IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="preamble",
                severity="high",
                summary="preamble axis — all positive so far",
                evidence=[
                    IssueEvidence(
                        source="f1.mb",
                        feedback="output starts directly with the list",
                        polarity="positive",
                    ),
                ],
            ),
        ],
    )
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )
    llm = MockLLMClient()
    results = calibrate(eval_file, feedback_list, issue_file, llm)
    assert results == []
    assert llm.calls == []


def test_calibrate_labels_uncited_entries_as_pass():
    """Entries reviewed but not cited as evidence are labeled PASS (holistic review)."""
    feedback_list = [
        _make_feedback("neg1.mb", "adds conversational preamble"),
        _make_feedback("unrelated.mb", "different problem entirely"),
    ]
    issue_file = _make_issue_file(["neg1.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # neg1=FAIL(TP), unrelated=PASS(TN — reviewed, not cited for this issue)
    llm = MockLLMClient(responses=[_FAIL_RESP, _PASS_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.num_examples == 2
    assert cal.accuracy == 1.0
    assert len(llm.calls) == 2
    labels = sorted(ex.label for ex in cal.examples)
    assert labels == ["FAIL", "PASS"]


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

    # 6 LLM calls: 3 entries × 2 evals. Each eval labels its evidence source
    # as FAIL and the other two as PASS (holistic review).
    llm = MockLLMClient(responses=[
        # eval-clarity-01: f1=FAIL(TP), f2=PASS(TN), f3=PASS(TN)
        _FAIL_RESP, _PASS_RESP, _PASS_RESP,
        # eval-completeness-02: f1=PASS(TN), f2=FAIL(TP), f3=PASS(TN)
        _PASS_RESP, _FAIL_RESP, _PASS_RESP,
    ])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    assert len(results) == 2
    assert results[0].eval_id == "eval-clarity-01"
    assert results[0].accuracy == 1.0
    assert results[0].num_examples == 3
    assert results[1].eval_id == "eval-completeness-02"
    assert results[1].accuracy == 1.0
    assert results[1].num_examples == 3
    assert len(llm.calls) == 6


def test_calibrate_matches_semicolon_split_entries():
    """Entries whose fragments appear in evidence (after ';' splitting) are matched."""
    # Reviewer wrote a compound observation; issue consolidation stored only
    # the relevant fragment as evidence. Calibrator must still match.
    feedback_list = [
        Feedback(
            source_file="run.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(
                    text="followed prompt; adds conversational preamble; face looks fine",
                ),
            ],
        ),
    ]
    issue_file = _make_issue_file(["run.mb"])  # evidence text: "adds conversational preamble"
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    llm = MockLLMClient(responses=[_FAIL_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    assert len(results) == 1
    cal = results[0]
    assert cal.num_examples == 1
    assert cal.examples[0].label == "FAIL"
    assert cal.accuracy == 1.0
    assert len(llm.calls) == 1


def test_calibrate_labels_entries_per_eval_independently():
    """A file's entries each get FAIL or PASS per eval based on that eval's evidence."""
    # One .mb file with 3 entries, only one cited as evidence for this eval.
    # The other two are PASS labels — reviewed but not flagged for this issue.
    feedback_list = [
        Feedback(
            source_file="review.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(text="adds conversational preamble"),   # evidence → FAIL
                FeedbackEntry(text="grammar was incorrect"),          # PASS for this issue
                FeedbackEntry(text="formatting looks off"),           # PASS for this issue
            ],
        ),
    ]
    issue_file = _make_issue_file(["review.mb"])
    eval_file = EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[_make_eval()],
    )

    # entry1=FAIL(TP), entry2=PASS(TN), entry3=PASS(TN)
    llm = MockLLMClient(responses=[_FAIL_RESP, _PASS_RESP, _PASS_RESP])

    results = calibrate(eval_file, feedback_list, issue_file, llm)
    cal = results[0]
    assert cal.num_examples == 3
    assert sorted(ex.label for ex in cal.examples) == ["FAIL", "PASS", "PASS"]
    assert cal.accuracy == 1.0
    assert len(llm.calls) == 3


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

def test_recall_zero_when_all_missed():
    """Recall is 0 when eval misses every known-bad example."""
    examples = [
        CalibrationExample(source="a.mb", label="FAIL", eval_result="PASS", match=False),
        CalibrationExample(source="b.mb", label="FAIL", eval_result="PASS", match=False),
    ]
    accuracy, precision, recall, f1, fp, fn = compute_metrics(examples)
    assert accuracy == 0.0
    assert recall == 0.0
    assert f1 == 0.0
    assert fp == 0
    assert fn == 2
