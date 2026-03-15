"""Tuning loop engine - orchestrates improve→test→improve iterations."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from prompterator.core.eval_runner import run_all_evals, save_result_file
from prompterator.core.improver import (
    generate_improved_prompt_with_rationale,
    generate_help_request,
    consolidate_redundant_lines,
)
from prompterator.runners.llm import debug_context
from prompterator.models.eval import EvalFile
from prompterator.models.issue import IssueFile
from prompterator.models.iteration import IterationRecord, PromptDiff, TuneReport
from prompterator.models.result import ResultFile, ResultSummary
from prompterator.runners.llm import LLMClient


def _run_evals_on_text(
    prompt_text: str,
    eval_file: EvalFile,
    critic_llm: LLMClient | None,
    *,
    author_llm: LLMClient | None = None,
    content_texts: list[str | None] | None = None,
    samples: int = 1,
    ensemble: int = 5,
    confidence_threshold: float = 9.0,
    script: str | None = None,
    script_timeout: int = 60,
    content_eval_map: dict[int, list[str]] | None = None,
) -> tuple[list, ResultSummary]:
    """Generate output from prompt text and evaluate it.

    Returns:
        Tuple of (eval_results, summary).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".prompt.txt", delete=False) as f:
        f.write(prompt_text)
        tmp_path = Path(f.name)

    try:
        result_file = run_all_evals(
            eval_file, tmp_path, critic_llm,
            author_llm=author_llm,
            content_texts=content_texts,
            samples=samples,
            ensemble=ensemble,
            confidence_threshold=confidence_threshold,
            script=script, script_timeout=script_timeout,
            content_eval_map=content_eval_map,
        )
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
    critic_llm: LLMClient | None = None,
    max_iterations: int = 20,
    on_iteration: Callable | None = None,
    on_status: Callable[[str], None] | None = None,
    *,
    author_llm: LLMClient | None = None,
    content_texts: list[str | None] | None = None,
    samples: int = 1,
    ensemble: int = 5,
    confidence_threshold: float = 9.0,
    critic_script: str | None = None,
    critic_script_timeout: int = 60,
    patience: int = 5,
    early_stop: bool = False,
    results_dir: Path | None = None,
    content_eval_map: dict[int, list[str]] | None = None,
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
        patience: Number of non-improving iterations to tolerate before stopping.

    Returns:
        TuneReport with all iteration records and final state.
    """
    with open(prompt_path) as f:
        original_text = f.read()

    current_text = original_text
    best_text = original_text
    iterations: list[IterationRecord] = []
    edit_history: list[dict] = []
    base_name = prompt_path.stem.split(".")[0]

    # Set up results run directory
    if results_dir is not None:
        from prompterator.core.run import create_run_dir

        run_dir = create_run_dir(results_dir)
    else:
        run_dir = None

    # Run baseline evals
    debug_context("tune.baseline.eval")
    baseline_results, baseline_summary = _run_evals_on_text(
        current_text, eval_file, critic_llm,
        author_llm=author_llm,
        content_texts=content_texts,
        samples=samples, ensemble=ensemble,
        confidence_threshold=confidence_threshold,
        script=critic_script, script_timeout=critic_script_timeout,
        content_eval_map=content_eval_map,
    )
    previous_results = baseline_results
    best_score = baseline_summary.overall_score
    best_results = baseline_results
    stall_count = 0

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    for i in range(1, max_iterations + 1):
        _status(f"Iteration {i}/{max_iterations}: consolidating...")
        # Consolidate redundant lines before generating a new edit
        consolidated, con_rationale, _con_raw = consolidate_redundant_lines(
            current_text, editor_llm,
        )
        if consolidated != current_text:
            _status(f"Iteration {i}/{max_iterations}: consolidated — {con_rationale}")
            current_text = consolidated
            best_text = consolidated
            with open(prompt_path, "w") as f:
                f.write(consolidated)

        # Generate improvement — retry up to 3 times if the edit doesn't produce a change
        max_retries = 3
        improved_text = current_text
        rationale = ""
        raw_output = ""
        edit_action = "UNKNOWN"

        for attempt in range(max_retries):
            _status(f"Iteration {i}/{max_iterations}: generating idea" + (f" (retry {attempt + 1})" if attempt else "") + "...")
            debug_context(f"tune.{i}.improve" + (f".retry{attempt}" if attempt else ""))
            improved_text, rationale, raw_output, edit_action = generate_improved_prompt_with_rationale(
                current_text, issue_file, editor_llm,
                eval_results=previous_results, iteration=i,
                edit_history=edit_history,
                stall_count=stall_count,
            )
            if improved_text != current_text:
                break
            _status(f"Iteration {i}/{max_iterations}: edit rejected — {rationale[:60]}")
            # Record the failed attempt in history so next retry knows
            edit_history.append({
                "rationale": rationale,
                "action": edit_action,
                "accepted": False,
            })

        # If all retries failed to produce a change, count as stall
        if improved_text == current_text:
            stall_count += 1
            if on_iteration:
                n_passed = sum(1 for r in previous_results if r.passed)
                n_total = len(previous_results)
                if n_passed == n_total:
                    v = "PASS"
                elif n_passed == 0 and best_score <= 2.5:
                    v = "FAIL"
                else:
                    v = "PARTIAL"
                record = IterationRecord(
                    iteration=i,
                    prompt_text=current_text,
                    rationale=rationale + " (no valid edit)",
                    diff=PromptDiff(before=current_text, after=current_text),
                    eval_results=previous_results,
                    summary=ResultSummary(
                        verdict=v, overall_score=best_score,
                        passed=n_passed, total=n_total,
                    ),
                    metric_deltas={},
                    l2_output=raw_output,
                )
                on_iteration(record)
            # Bail out if stuck for too long — no valid edits possible
            if stall_count >= max(patience, 3):
                _status(f"Bailing out — stuck for {stall_count} iterations")
                break
            continue

        _status(f"Iteration {i}/{max_iterations}: evaluating...")

        # Build diff
        diff = PromptDiff(before=current_text, after=improved_text)

        # Run evals on improved prompt
        debug_context(f"tune.{i}.eval")
        new_results, new_summary = _run_evals_on_text(
            improved_text, eval_file, critic_llm,
            author_llm=author_llm,
            content_texts=content_texts,
            samples=samples, ensemble=ensemble,
            confidence_threshold=confidence_threshold,
            script=critic_script, script_timeout=critic_script_timeout,
            content_eval_map=content_eval_map,
        )

        # Compute deltas
        metric_deltas = _compute_metric_deltas(new_results, best_results)

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

        # Save per-iteration results and prompt snapshot
        if run_dir is not None:
            result_file = ResultFile(
                prompt_tested=str(prompt_path),
                results=new_results,
                summary=new_summary,
            )
            save_result_file(result_file, run_dir / f"{base_name}.{i:03d}.results.yaml")
            snapshot_path = run_dir / f"{base_name}.{i:03d}.prompt{prompt_path.suffix}"
            snapshot_path.write_text(improved_text)

        # Accept or reject: only move forward if score improved
        accepted = new_summary.overall_score >= best_score
        edit_history.append({
            "rationale": rationale,
            "action": edit_action,
            "accepted": accepted,
        })

        # Use a noise margin so small score fluctuations from low sample
        # counts don't cause false stalls or false improvements.
        noise_margin = 5.0 / max(len(new_results), 1)

        if new_summary.overall_score > best_score + noise_margin:
            # Clear improvement beyond noise
            best_score = new_summary.overall_score
            best_text = improved_text
            best_results = new_results
            current_text = improved_text
            previous_results = new_results
            stall_count = 0

            # Write improved prompt to source file
            with open(prompt_path, "w") as f:
                f.write(improved_text)
        elif new_summary.overall_score >= best_score - noise_margin:
            # Within noise band — accept to explore but don't update best
            current_text = improved_text
            previous_results = new_results
            if new_summary.overall_score > best_score:
                best_score = new_summary.overall_score
                best_text = improved_text
                best_results = new_results
                with open(prompt_path, "w") as f:
                    f.write(improved_text)
            stall_count += 1
        else:
            # Clear regression — revert to best prompt
            current_text = best_text
            previous_results = best_results
            stall_count += 1

        # Check termination: all pass (only in early_stop mode)
        if early_stop and new_summary.verdict == "PASS":
            best_text = improved_text
            best_results = new_results
            break

        # Check termination: patience exhausted (only in early_stop mode)
        if early_stop and stall_count >= patience and i > 1:
            break

    # Use the best prompt tracked during the loop
    final_text = best_text
    final_results = best_results
    # Find the summary for the best results
    if best_results is baseline_results:
        final_summary = baseline_summary
    else:
        best_iter = [r for r in iterations if r.prompt_text == best_text]
        final_summary = best_iter[-1].summary if best_iter else baseline_summary

    metric_table = _build_metric_table(baseline_results, final_results)

    # Generate help request if we plateaued or got stuck without passing
    help_request = None
    if final_summary.verdict != "PASS" and stall_count >= min(patience, 3):
        debug_context("tune.help-request")
        help_request = generate_help_request(
            best_text, issue_file, editor_llm,
            eval_results=best_results,
            edit_history=edit_history,
        )

    report = TuneReport(
        prompt_ref=str(prompt_path),
        max_iterations=max_iterations,
        iterations=iterations,
        final_prompt=final_text,
        final_summary=final_summary,
        metric_table=metric_table,
        help_request=help_request,
    )

    # Save outputs
    if run_dir is not None:
        import yaml

        report_path = run_dir / "tune-report.yaml"
        with open(report_path, "w") as f:
            yaml.dump(report.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)

    # Always write the best prompt back to the source file
    with open(prompt_path, "w") as f:
        f.write(final_text)

    return report
