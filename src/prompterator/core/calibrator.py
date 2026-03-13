"""Calibration logic - verify evals agree with human labels."""

from __future__ import annotations

from pathlib import Path

from prompterator.models.calibration import (
    CalibrationExample,
    CalibrationReport,
    CalibrationResult,
)
from prompterator.models.eval import Eval, EvalFile
from prompterator.models.feedback import Feedback
from prompterator.models.issue import IssueFile
from prompterator.runners.llm import LLMClient


def classify_labels(
    feedback_list: list[Feedback],
    issue_file: IssueFile,
    eval_spec: Eval,
) -> dict[str, str]:
    """Classify feedback files as PASS or FAIL relative to an eval.

    Uses the heuristic approach: files that appear in the issue's evidence
    (linked via the eval's issue_ref) are labelled FAIL (negative);
    all other feedback files are labelled PASS (positive).

    Args:
        feedback_list: All parsed feedback objects for the prompt.
        eval_spec: The eval to classify against.
        issue_file: The issue file containing evidence mappings.

    Returns:
        Dict mapping source file basename to "PASS" or "FAIL".
    """
    # Find the issue linked to this eval
    negative_sources: set[str] = set()

    if eval_spec.issue_ref:
        for issue in issue_file.issues:
            if issue.id == eval_spec.issue_ref:
                for ev in issue.evidence:
                    negative_sources.add(Path(ev.source).name)
                break

    labels: dict[str, str] = {}
    for fb in feedback_list:
        name = Path(fb.source_file).name
        labels[name] = "FAIL" if name in negative_sources else "PASS"

    return labels


def _build_output_rubric_prompt(
    feedback_text: str,
    criteria: list[str],
) -> str:
    """Build a prompt for output-level rubric evaluation.

    Instead of asking whether a *prompt* addresses criteria, this asks
    whether a specific *output* (represented by its human feedback)
    exhibits the problems described by the criteria.
    """
    criteria_list = "\n".join(f"- {c}" for c in criteria)
    return f"""You are evaluating whether a specific output exhibits certain problems.

The output was reviewed by a human who provided the following feedback:

FEEDBACK:
---
{feedback_text}
---

Evaluate this feedback against the following criteria. For each criterion,
determine whether the feedback indicates the output HAS the problem
described (FAIL) or does NOT have the problem (PASS).

Criteria:
{criteria_list}

Respond in this format for each criterion:
CRITERION: [criterion text]
RESULT: PASS or FAIL
REASON: [brief explanation]

After evaluating all criteria, provide:
OVERALL: PASS (if all criteria pass) or FAIL (if any fail)
SCORE: [0.0-1.0 based on pass rate]"""


def run_calibration_eval(
    eval_spec: Eval,
    feedback_text: str,
    llm_client: LLMClient,
) -> bool:
    """Run a single eval against a feedback example's text.

    Uses output-level evaluation: asks the LLM whether the feedback
    indicates the output exhibits the problem described by the eval
    criteria.

    Args:
        eval_spec: The eval specification.
        feedback_text: Combined feedback text from the .mb file.
        llm_client: LLM client for critic calls.

    Returns:
        True if eval passes (output does NOT have the problem), False otherwise.
    """
    if eval_spec.type != "rubric" or not eval_spec.rubric:
        return True

    criteria = eval_spec.rubric.criteria
    prompt = _build_output_rubric_prompt(feedback_text, criteria)
    system = "You are an expert evaluator. Be objective and thorough."
    response = llm_client.generate(prompt, system=system, temperature=0.3)

    # Parse response - reuse logic from eval_runner
    from prompterator.core.eval_runner import _parse_rubric_response

    passed, _score, _details = _parse_rubric_response(response, criteria)
    return passed


def compute_metrics(
    examples: list[CalibrationExample],
) -> tuple[float, float, float, float, int, int]:
    """Compute calibration metrics from examples.

    Convention: "positive" in the confusion-matrix sense means the eval
    detected a problem (FAIL).  So:
      - TP = eval FAIL and label FAIL (correctly detected problem)
      - TN = eval PASS and label PASS (correctly found no problem)
      - FP = eval FAIL but label PASS (false alarm)
      - FN = eval PASS but label FAIL (missed problem)

    Returns:
        Tuple of (accuracy, precision, recall, f1, false_positives, false_negatives).
    """
    tp = sum(1 for e in examples if e.eval_result == "FAIL" and e.label == "FAIL")
    tn = sum(1 for e in examples if e.eval_result == "PASS" and e.label == "PASS")
    fp = sum(1 for e in examples if e.eval_result == "FAIL" and e.label == "PASS")
    fn = sum(1 for e in examples if e.eval_result == "PASS" and e.label == "FAIL")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, fp, fn


def determine_verdict(accuracy: float) -> str:
    """Determine calibration verdict from accuracy.

    Returns:
        "GOOD" (>= 0.80), "WEAK" (>= 0.60), or "BAD" (< 0.60).
    """
    if accuracy >= 0.80:
        return "GOOD"
    elif accuracy >= 0.60:
        return "WEAK"
    else:
        return "BAD"


def calibrate(
    eval_file: EvalFile,
    feedback_list: list[Feedback],
    issue_file: IssueFile,
    llm_client: LLMClient,
) -> list[CalibrationResult]:
    """Run calibration for all evals against labeled feedback.

    For each eval, classifies feedback as positive/negative using issue
    evidence, runs the eval against each feedback example, and computes
    agreement metrics.

    Args:
        eval_file: The eval specifications to calibrate.
        feedback_list: All parsed feedback for the prompt.
        issue_file: Issue file for label classification.
        llm_client: LLM client for running evals.

    Returns:
        List of CalibrationResult, one per eval.
    """
    results: list[CalibrationResult] = []

    for eval_spec in eval_file.evals:
        labels = classify_labels(feedback_list, issue_file, eval_spec)

        examples: list[CalibrationExample] = []
        for fb in feedback_list:
            name = Path(fb.source_file).name
            label = labels.get(name, "PASS")

            # Combine all feedback entries into a single text block
            combined_text = " ".join(entry.text for entry in fb.entries)
            if not combined_text.strip():
                continue

            eval_passed = run_calibration_eval(eval_spec, combined_text, llm_client)
            eval_result = "PASS" if eval_passed else "FAIL"

            examples.append(
                CalibrationExample(
                    source=name,
                    label=label,
                    eval_result=eval_result,
                    match=(label == eval_result),
                )
            )

        if not examples:
            continue

        accuracy, precision, recall, f1, fp, fn = compute_metrics(examples)
        verdict = determine_verdict(accuracy)

        results.append(
            CalibrationResult(
                eval_id=eval_spec.id,
                num_examples=len(examples),
                accuracy=round(accuracy, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                false_positives=fp,
                false_negatives=fn,
                verdict=verdict,
                examples=examples,
            )
        )

    return results


def save_calibration_report(report: CalibrationReport, path: Path) -> None:
    """Save a calibration report to disk as YAML."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(report.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)
