"""Tuning loop engine - orchestrates improve→test→improve iterations."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from prompterator.core.eval_runner import run_all_evals
from prompterator.core.improver import generate_improved_prompt_with_rationale
from prompterator.models.eval import EvalFile
from prompterator.models.issue import IssueFile
from prompterator.models.iteration import IterationRecord, PromptDiff, TuneReport
from prompterator.models.result import ResultSummary
from prompterator.runners.llm import LLMClient


def _run_evals_on_text(
    prompt_text: str,
    eval_file: EvalFile,
    critic_llm: LLMClient,
) -> tuple[list, ResultSummary]:
    """Run evals on prompt text by writing to a temp file.

    Returns:
        Tuple of (eval_results, summary).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".prompt.txt", delete=False) as f:
        f.write(prompt_text)
        tmp_path = Path(f.name)

    try:
        result_file = run_all_evals(eval_file, tmp_path, critic_llm)
    finally:
        tmp_path.unlink(missing_ok=True)

    return result_file.results, result_file.summary


def _compute_metric_deltas(
    current_results: list,
    previous_results: list,
) -> dict[str, float]:
    """Compute per-eval score deltas between iterations."""
    prev_scores = {r.eval_id: r.score for r in previous_results}
    deltas = {}
    for r in current_results:
        prev = prev_scores.get(r.eval_id, 0.0)
        deltas[r.eval_id] = round(r.score - prev, 4)
    return deltas


def _build_metric_table(
    baseline_results: list,
    final_results: list,
) -> list[dict]:
    """Build a metric table comparing baseline to final results."""
    baseline_scores = {r.eval_id: r.score for r in baseline_results}
    table = []
    for r in final_results:
        before = baseline_scores.get(r.eval_id, 0.0)
        table.append({
            "eval_id": r.eval_id,
            "before": round(before, 4),
            "after": round(r.score, 4),
            "delta": round(r.score - before, 4),
        })
    return table


def run_tuning_loop(
    prompt_path: Path,
    issue_file: IssueFile,
    eval_file: EvalFile,
    editor_llm: LLMClient,
    critic_llm: LLMClient,
    max_iterations: int = 20,
    output_dir: Path | None = None,
    on_iteration: Callable | None = None,
) -> TuneReport:
    """Run the full tuning loop.

    Args:
        prompt_path: Path to the original prompt file.
        issue_file: Issues to address.
        eval_file: Evals to run.
        editor_llm: LLM client for prompt improvement.
        critic_llm: LLM client for evaluation.
        max_iterations: Maximum number of iterations.
        output_dir: Directory for output files.
        on_iteration: Optional callback(iteration_record) for progress reporting.

    Returns:
        TuneReport with all iteration records and final state.
    """
    with open(prompt_path) as f:
        original_text = f.read()

    current_text = original_text
    iterations: list[IterationRecord] = []

    # Run baseline evals
    baseline_results, baseline_summary = _run_evals_on_text(
        current_text, eval_file, critic_llm
    )
    previous_results = baseline_results
    previous_score = baseline_summary.overall_score

    for i in range(1, max_iterations + 1):
        # Generate improvement
        improved_text, rationale, raw_output = generate_improved_prompt_with_rationale(
            current_text, issue_file, editor_llm,
            eval_results=previous_results, iteration=i,
        )

        # Build diff
        diff = PromptDiff(before=current_text, after=improved_text)

        # Run evals on improved prompt
        new_results, new_summary = _run_evals_on_text(
            improved_text, eval_file, critic_llm
        )

        # Compute deltas
        metric_deltas = _compute_metric_deltas(new_results, previous_results)

        # Build iteration record
        record = IterationRecord(
            iteration=i,
            prompt_text=improved_text,
            rationale=rationale,
            diff=diff,
            eval_results=new_results,
            summary=new_summary,
            metric_deltas=metric_deltas,
            l2_output=raw_output,
        )
        iterations.append(record)

        if on_iteration:
            on_iteration(record)

        # Check termination: no improvement
        if new_summary.overall_score <= previous_score and i > 1:
            break

        # Check termination: all pass
        if new_summary.verdict == "PASS":
            current_text = improved_text
            previous_results = new_results
            previous_score = new_summary.overall_score
            break

        # Continue
        current_text = improved_text
        previous_results = new_results
        previous_score = new_summary.overall_score

    # Use the best iteration's prompt as final
    if iterations:
        best = max(iterations, key=lambda r: r.summary.overall_score)
        final_text = best.prompt_text
        final_summary = best.summary
        final_results = best.eval_results
    else:
        final_text = original_text
        final_summary = baseline_summary
        final_results = baseline_results

    metric_table = _build_metric_table(baseline_results, final_results)

    report = TuneReport(
        prompt_ref=str(prompt_path),
        max_iterations=max_iterations,
        iterations=iterations,
        final_prompt=final_text,
        final_summary=final_summary,
        metric_table=metric_table,
    )

    # Save outputs if output_dir specified
    if output_dir:
        import yaml

        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "tune-report.yaml"
        with open(report_path, "w") as f:
            yaml.dump(report.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)

        final_path = output_dir / prompt_path.name
        with open(final_path, "w") as f:
            f.write(final_text)

    return report
