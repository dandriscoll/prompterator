"""Calibration logic - verify evals agree with human labels."""

from __future__ import annotations

from pathlib import Path

from prompterator.core.issue import _split_feedback_entry
from prompterator.core.progress import Progress
from prompterator.models.calibration import (
    CalibrationExample,
    CalibrationReport,
    CalibrationResult,
)
from prompterator.models.eval import Eval, EvalFile
from prompterator.models.feedback import Feedback, FeedbackEntry
from prompterator.models.issue import IssueFile
from prompterator.runners.llm import LLMClient


def _entry_matches_evidence(
    mb_name: str,
    src_name: str,
    entry_text: str,
    evidence_entries: set[tuple[str, str]],
) -> bool:
    """True if the entry matches any (source, feedback) pair in evidence.

    Issue consolidation applies ``_split_feedback_entry`` to entries before
    storing them as evidence, so a compound entry like "A; B; C" can appear
    in evidence as the fragment "B". Match on the full entry text first, then
    on each split part, so calibrator and issue generator stay consistent.
    """
    if (mb_name, entry_text) in evidence_entries or (src_name, entry_text) in evidence_entries:
        return True
    for part in _split_feedback_entry(entry_text):
        if part == entry_text:
            continue
        if (mb_name, part) in evidence_entries or (src_name, part) in evidence_entries:
            return True
    return False


def classify_labels(
    feedback_list: list[Feedback],
    issue_file: IssueFile,
    eval_spec: Eval,
) -> set[tuple[str, str]]:
    """Identify specific feedback entries that are evidence for an eval's issue.

    Matches at the entry level: an entry is labeled FAIL only if its text
    appears in the linked issue's evidence. Other entries from the same
    .mb file but about different categories are not included.

    Args:
        feedback_list: All parsed feedback objects for the prompt.
        eval_spec: The eval to classify against.
        issue_file: The issue file containing evidence mappings.

    Returns:
        Set of (source_basename, feedback_text) tuples that are known FAIL.
        Entries not in the set were not assessed and should be skipped.
    """
    # Collect evidence texts keyed by source basename
    evidence_entries: set[tuple[str, str]] = set()

    if eval_spec.issue_ref:
        for issue in issue_file.issues:
            if issue.id == eval_spec.issue_ref:
                for ev in issue.evidence:
                    evidence_entries.add((Path(ev.source).name, ev.feedback))
                break

    return evidence_entries


def _build_output_rubric_prompt(
    feedback_text: str,
    criteria: list[str],
    *,
    input_content: str | None = None,
) -> str:
    """Build a prompt for output-level rubric evaluation.

    Instead of asking whether a *prompt* addresses criteria, this asks
    whether a specific *output* (represented by its human feedback)
    exhibits the problems described by the criteria.
    """
    criteria_list = "\n".join(f"- {c}" for c in criteria)

    input_section = ""
    if input_content:
        input_section = f"""
The output was generated from this input:

INPUT:
---
{input_content}
---

"""

    return f"""You are evaluating whether a specific output exhibits certain problems.
{input_section}
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
    *,
    input_content: str | None = None,
) -> bool:
    """Run a single eval against a feedback example's text.

    Uses output-level evaluation: asks the LLM whether the feedback
    indicates the output exhibits the problem described by the eval
    criteria.

    Args:
        eval_spec: The eval specification.
        feedback_text: Feedback text from a single .mb record.
        llm_client: LLM client for critic calls.
        input_content: The original input/content the author was given.

    Returns:
        True if eval passes (output does NOT have the problem), False otherwise.
    """
    if eval_spec.type != "rubric" or not eval_spec.rubric:
        return True

    criteria = eval_spec.rubric.criteria
    prompt = _build_output_rubric_prompt(
        feedback_text, criteria, input_content=input_content,
    )
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

    Examples may carry either FAIL labels (entry cited as evidence for
    this eval's issue) or PASS labels (entry reviewed but not cited —
    i.e. the reviewer examined this output and did not flag the problem).

      - TP = eval FAIL and label FAIL (correctly caught problem)
      - FP = eval FAIL but label PASS (false alarm on clean output)
      - FN = eval PASS but label FAIL (missed problem)
      - TN = eval PASS and label PASS (correctly cleared clean output)

    Precision defaults to 1.0 when no eval FAIL predictions exist;
    recall defaults to 1.0 when no FAIL labels exist.

    Returns:
        Tuple of (accuracy, precision, recall, f1, false_positives, false_negatives).
    """
    tp = sum(1 for e in examples if e.eval_result == "FAIL" and e.label == "FAIL")
    fp = sum(1 for e in examples if e.eval_result == "FAIL" and e.label == "PASS")
    fn = sum(1 for e in examples if e.eval_result == "PASS" and e.label == "FAIL")
    tn = sum(1 for e in examples if e.eval_result == "PASS" and e.label == "PASS")

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, fp, fn


def determine_verdict(precision: float, recall: float) -> str:
    """Determine calibration verdict from precision and recall.

    Taking the worst of the two surfaces both failure modes:
      - low recall  → criteria too narrow (eval misses real problems)
      - low precision → criteria too broad (eval fires on clean outputs)

    Returns:
        "GOOD" (both >= 0.80), "WEAK" (both >= 0.60), or "BAD" (either < 0.60).
    """
    worst = min(precision, recall)
    if worst >= 0.80:
        return "GOOD"
    elif worst >= 0.60:
        return "WEAK"
    else:
        return "BAD"


def _resolve_input_content(
    input_ref: str | None,
    feedback_dir: Path | None,
) -> str | None:
    """Try to read the input/content file referenced by an input_ref.

    Returns the file contents if found, None otherwise.
    """
    if not input_ref:
        return None

    ref_path = Path(input_ref)
    # Try as-is (absolute or relative to cwd)
    if ref_path.exists():
        return ref_path.read_text()
    # Try relative to feedback directory
    if feedback_dir is not None:
        candidate = feedback_dir / ref_path
        if candidate.exists():
            return candidate.read_text()
    return None


def estimate_calibration_calls(
    eval_file: EvalFile,
    feedback_list: list[Feedback],
    issue_file: IssueFile,
) -> int:
    """Estimate total LLM calls for calibration.

    For each eval with evidence, every reviewed entry becomes a calibration
    example: entries matching the eval's evidence are FAIL labels, the rest
    are implicit PASS labels (reviewed but uncited for this issue).
    """
    reviewed_per_eval = sum(
        1
        for fb in feedback_list
        for entry in fb.entries
        if entry.text.strip()
    )
    total = 0
    for eval_spec in eval_file.evals:
        evidence_entries = classify_labels(feedback_list, issue_file, eval_spec)
        if not evidence_entries:
            continue
        total += reviewed_per_eval
    return total


def calibrate(
    eval_file: EvalFile,
    feedback_list: list[Feedback],
    issue_file: IssueFile,
    llm_client: LLMClient,
    *,
    feedback_dir: Path | None = None,
    progress: Progress | None = None,
) -> list[CalibrationResult]:
    """Run calibration for all evals against labeled feedback.

    Each markback record is treated as an individual calibration row.
    The record's file_ref identifies the output, and input_ref
    identifies the input content — both are used for label matching
    and passed to the eval for alignment checking.

    Args:
        eval_file: The eval specifications to calibrate.
        feedback_list: All parsed feedback for the prompt.
        issue_file: Issue file for label classification.
        llm_client: LLM client for running evals.
        feedback_dir: Directory containing .mb files (for resolving input refs).

    Returns:
        List of CalibrationResult, one per eval.
    """
    # Flatten once — every non-empty entry across all feedback files counts
    # as "reviewed" and produces a calibration example for each eval that
    # has any evidence. Labels are per-eval: FAIL if cited as evidence for
    # the eval's issue, PASS otherwise (holistic-review assumption).
    reviewed: list[tuple[str, Path, FeedbackEntry, str]] = []
    for fb in feedback_list:
        mb_name = Path(fb.source_file).name
        mb_dir = Path(fb.source_file).parent
        for entry in fb.entries:
            if not entry.text.strip():
                continue
            src_name = Path(entry.file_ref).name if entry.file_ref else mb_name
            reviewed.append((mb_name, mb_dir, entry, src_name))

    results: list[CalibrationResult] = []

    for eval_spec in eval_file.evals:
        evidence_entries = classify_labels(feedback_list, issue_file, eval_spec)

        if not evidence_entries:
            continue

        examples: list[CalibrationExample] = []
        for mb_name, mb_dir, entry, src_name in reviewed:
            if _entry_matches_evidence(mb_name, src_name, entry.text, evidence_entries):
                label = "FAIL"
            else:
                label = "PASS"

            input_content = _resolve_input_content(
                entry.input_ref, feedback_dir or mb_dir,
            )

            eval_passed = run_calibration_eval(
                eval_spec, entry.text, llm_client,
                input_content=input_content,
            )
            if progress:
                progress.tick(eval_spec.id)
            eval_result = "PASS" if eval_passed else "FAIL"

            examples.append(
                CalibrationExample(
                    source=src_name,
                    label=label,
                    eval_result=eval_result,
                    match=(label == eval_result),
                )
            )

        if not examples:
            continue

        accuracy, precision, recall, f1, fp, fn = compute_metrics(examples)
        verdict = determine_verdict(precision, recall)

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


_REVISE_CRITERIA_SYSTEM = """\
You are an eval criteria editor. You will see an eval's rubric criteria and \
examples where the eval disagreed with a human reviewer. Your job is to \
revise the criteria so they agree with the human.

You may see:
- MISSED (false negatives): the eval said PASS but the human said FAIL \
— the criteria are too narrow or missing an important check. Broaden them.
- FALSE ALARMS (false positives): the eval said FAIL but the human said PASS \
— the criteria are too broad and flag clean outputs. Narrow them.

Respond with ONLY a JSON array of revised criterion strings. \
Keep the same number of criteria or fewer. Do not add markdown fences.

Rules:
- Balance breadth: catch real problems without flagging clean outputs
- If a criterion is fundamentally wrong, replace it entirely
- Frame criteria as what the output must NOT do (absence of problem = PASS)
- Each criterion should be one clear, testable sentence"""


def revise_eval_criteria(
    eval_spec: Eval,
    calibration: CalibrationResult,
    feedback_list: list[Feedback],
    llm_client: LLMClient,
) -> list[str] | None:
    """Use calibration mismatches to revise eval criteria.

    Args:
        eval_spec: The eval whose criteria need revision.
        calibration: The calibration result showing mismatches.
        feedback_list: All parsed feedback (to look up text for examples).
        llm_client: LLM client for generating revised criteria.

    Returns:
        Revised criteria list, or None if no revision needed.
    """
    if eval_spec.type != "rubric" or not eval_spec.rubric:
        return None

    # Build lookup: source basename -> feedback text
    fb_by_source: dict[str, list[str]] = {}
    for fb in feedback_list:
        for entry in fb.entries:
            src = Path(entry.file_ref).name if entry.file_ref else Path(fb.source_file).name
            fb_by_source.setdefault(src, []).append(entry.text)

    fn_examples: list[str] = []  # eval PASS, label FAIL — criteria too narrow
    fp_examples: list[str] = []  # eval FAIL, label PASS — criteria too broad
    for ex in calibration.examples:
        if ex.match:
            continue
        texts = fb_by_source.get(ex.source, [])
        text_block = "; ".join(texts) if texts else "(no feedback text available)"
        line = f"- {ex.source}: {text_block}"
        if ex.eval_result == "PASS" and ex.label == "FAIL":
            fn_examples.append(line)
        elif ex.eval_result == "FAIL" and ex.label == "PASS":
            fp_examples.append(line)

    if not fn_examples and not fp_examples:
        return None

    criteria_text = "\n".join(f"- {c}" for c in eval_spec.rubric.criteria)

    parts = [f"EVAL: {eval_spec.id}"]
    if eval_spec.description:
        parts.append(f"DESCRIPTION: {eval_spec.description}")
    parts.append(f"\nCURRENT CRITERIA:\n{criteria_text}")
    if fn_examples:
        parts.append(
            f"\nMISSED (eval said PASS, human said FAIL — "
            f"criteria too narrow):\n" + "\n".join(fn_examples)
        )
    if fp_examples:
        parts.append(
            f"\nFALSE ALARMS (eval said FAIL, human said PASS — "
            f"criteria too broad):\n" + "\n".join(fp_examples)
        )
    parts.append("\nRevise the criteria to agree with the human labels.")

    import json
    raw = llm_client.generate("\n".join(parts), system=_REVISE_CRITERIA_SYSTEM)
    try:
        revised = json.loads(raw)
        if isinstance(revised, list) and revised:
            return [str(c) for c in revised]
    except (json.JSONDecodeError, Exception):
        pass

    return None


def save_calibration_report(report: CalibrationReport, path: Path) -> None:
    """Save a calibration report to disk as YAML."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(report.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)
