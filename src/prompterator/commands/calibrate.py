"""Calibrate command - verify evals agree with human-labeled feedback."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.calibrator import calibrate, estimate_calibration_calls, revise_eval_criteria, save_calibration_report
from prompterator.core.progress import Progress
from prompterator.core.run import create_run_dir
from prompterator.commands.resolve import ResolveError, resolve_prompt_and_evals, resolve_issues
from prompterator.commands.feedback import find_mb_files, parse_mb_file
from prompterator.models.calibration import CalibrationReport
from prompterator.runners.llm import LLMClient, LLMError, enable_verbose


@click.command("calibrate")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option(
    "--evals",
    "evals_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to eval file (default: auto-detect from prompt name)",
)
@click.option(
    "--feedback-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing .mb files (default: from config)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for calibration report (default: auto-generated)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show per-example details",
)
@click.option(
    "--auto-fix",
    is_flag=True,
    help="Automatically revise WEAK/BAD eval criteria using LLM (default: suggest command)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress live progress output",
)
def calibrate_cmd(
    prompt: Path | None,
    evals_path: Path | None,
    feedback_dir: Path | None,
    output: Path | None,
    verbose: bool,
    auto_fix: bool,
    quiet: bool,
) -> None:
    """Verify that evals agree with human-labeled feedback.

    PROMPT is the path to the prompt file (optional — can be derived
    from eval/issue files).

    Runs each eval against the labeled feedback examples and reports
    whether the eval verdicts match the human labels. This validates
    that evals are reliable before using them to drive the tuning loop.

    When WEAK or BAD evals are found, prints a suggested
    ``prompterator evals -d`` command. Use --auto-fix to have the LLM
    revise criteria automatically instead.
    """
    if verbose:
        enable_verbose()

    config = load_config()
    base_dir = get_config_base_dir()

    try:
        prompt, evals_path, eval_file = resolve_prompt_and_evals(
            config, base_dir, prompt, evals_path,
        )
        issues_path, issue_file = resolve_issues(
            config, base_dir, prompt,
        )
    except ResolveError as e:
        click.echo(str(e))
        raise SystemExit(1)

    if not eval_file.evals:
        click.echo("No evaluations to calibrate.")
        raise SystemExit(1)

    # --- Load feedback files ----------------------------------------------
    if feedback_dir is None:
        feedback_dir = config.get_dir("feedback", base_dir)

    click.echo(f"Searching for .mb feedback files in {feedback_dir}...")
    mb_files = find_mb_files(feedback_dir)
    if not mb_files:
        click.echo(f"No .mb files found in {feedback_dir}")
        raise SystemExit(1)

    # Parse and filter feedback for this prompt
    prompt_ref = eval_file.prompt_ref
    feedback_list = []
    for path in mb_files:
        try:
            fb = parse_mb_file(path)
            # Keep feedback that references this prompt (or all if no ref)
            if fb.prompt_ref is None or Path(fb.prompt_ref).name == Path(prompt_ref).name:
                feedback_list.append(fb)
        except Exception as e:
            click.echo(f"Warning: could not parse {path}: {e}", err=True)

    if not feedback_list:
        click.echo(f"No feedback files found referencing {prompt_ref}")
        raise SystemExit(1)

    # --- Print header -----------------------------------------------------
    n_evals = len(eval_file.evals)
    n_feedback = len(feedback_list)
    n_entries = sum(len(fb.entries) for fb in feedback_list)
    n_llm = estimate_calibration_calls(eval_file, feedback_list, issue_file)

    click.echo()
    click.echo(f"Prompt:         {prompt_ref}")
    click.echo(f"Evals:          {n_evals} from {evals_path.name}")
    click.echo(f"Feedback:       {n_feedback} .mb files, {n_entries} records")
    click.echo(f"LLM calls:      {n_llm} (every reviewed entry × evals with evidence)")
    click.echo()

    # --- Initialize LLM ---------------------------------------------------
    try:
        llm = LLMClient(**config.resolve_role("critic"))
    except LLMError as e:
        click.echo(f"Critic LLM error: {e}", err=True)
        raise SystemExit(1)

    # --- Create run dir early so debug logs land in it --------------------
    base_name = prompt.stem.split(".")[0]
    if output is None:
        results_dir = config.get_dir("results", base_dir)
        run_dir = create_run_dir(results_dir)
        output = run_dir / f"{base_name}.calibration.yaml"

    # --- Run calibration --------------------------------------------------
    from prompterator.runners.llm import debug_context
    debug_context("calibrate")
    click.echo(
        "Running each eval against every reviewed entry to check whether\n"
        "the eval agrees with the human labels (FAIL where cited as evidence,\n"
        "PASS otherwise)..."
    )
    click.echo()
    progress = Progress(n_llm, label="Calibrating", quiet=quiet)
    try:
        cal_results = calibrate(eval_file, feedback_list, issue_file, llm,
                                feedback_dir=feedback_dir, progress=progress)
    except LLMError as e:
        click.echo(f"LLM error during calibration: {e}", err=True)
        raise SystemExit(1)
    finally:
        progress.finish()

    if not cal_results:
        click.echo("No calibration results produced.")
        raise SystemExit(1)

    # --- Display results --------------------------------------------------
    any_bad = False
    needs_fix = []

    for cal in cal_results:
        # Find the eval spec to show criteria
        eval_spec = next((ev for ev in eval_file.evals if ev.id == cal.eval_id), None)

        click.echo(f"Eval: {cal.eval_id}")
        if eval_spec and eval_spec.description:
            click.echo(f"  {eval_spec.description}")
        if eval_spec and eval_spec.rubric:
            for c in eval_spec.rubric.criteria:
                click.echo(f"  Criterion: {c}")

        if verbose:
            click.echo()
            header = f"  {'Example':<35} {'Label':<10} {'Eval':<10} {'Match'}"
            click.echo(header)
            click.echo("  " + "-" * (len(header) - 2))
            for ex in cal.examples:
                match_str = "yes" if ex.match else click.style("NO", fg="red", bold=True)
                click.echo(f"  {ex.source:<35} {ex.label:<10} {ex.eval_result:<10} {match_str}")
            click.echo("  " + "-" * (len(header) - 2))

        n_correct = int(round(cal.accuracy * cal.num_examples))
        click.echo(
            f"  Accuracy:  {n_correct}/{cal.num_examples} ({cal.accuracy * 100:.1f}%)"
        )
        click.echo(
            f"  Precision: {cal.precision * 100:.1f}%  "
            f"Recall: {cal.recall * 100:.1f}%  F1: {cal.f1:.2f}"
        )
        if cal.false_negatives > 0:
            click.echo(
                f"  Missed: {cal.false_negatives}"
                " (eval said PASS but human said FAIL — criteria too narrow)"
            )
        if cal.false_positives > 0:
            click.echo(
                f"  False alarms: {cal.false_positives}"
                " (eval said FAIL but human said PASS — criteria too broad)"
            )

        verdict_color = {"GOOD": "green", "WEAK": "yellow", "BAD": "red"}[cal.verdict]
        reason = {
            "GOOD": "precision and recall >= 80%",
            "WEAK": "precision or recall 60-79%",
            "BAD": "precision or recall < 60%",
        }[cal.verdict]
        click.echo(
            f"  Verdict: {click.style(cal.verdict, fg=verdict_color, bold=True)} ({reason})"
        )
        click.echo()

        if cal.verdict in ("WEAK", "BAD"):
            any_bad = True
            needs_fix.append(cal)

    # --- Save report ------------------------------------------------------
    report = CalibrationReport(
        prompt_ref=prompt_ref,
        eval_file=str(evals_path),
        calibrations=cal_results,
    )
    save_calibration_report(report, output)
    click.echo(f"Calibration report saved to: {output}")

    # --- Fix mode ---------------------------------------------------------
    if needs_fix and auto_fix:
        click.echo()
        click.echo(
            f"Revising {len(needs_fix)} eval(s) with WEAK/BAD detection.\n"
            f"Using missed examples to broaden criteria..."
        )

        # Use editor LLM for criteria revision
        try:
            editor_llm = LLMClient(**config.resolve_role("editor"))
        except LLMError as e:
            click.echo(f"Editor LLM error: {e}", err=True)
            raise SystemExit(1)

        # Build eval lookup
        eval_by_id = {ev.id: ev for ev in eval_file.evals}
        revised_count = 0

        for cal in needs_fix:
            eval_spec = eval_by_id.get(cal.eval_id)
            if not eval_spec or not eval_spec.rubric:
                continue

            click.echo()
            click.echo(f"  {cal.eval_id}:")
            click.echo(f"    Missed: {cal.false_negatives}")
            click.echo(f"    Current criteria:")
            for c in eval_spec.rubric.criteria:
                click.echo(f"      - {c}")

            click.echo(f"    Asking LLM to revise criteria...")

            revised = revise_eval_criteria(
                eval_spec, cal, feedback_list, editor_llm,
            )

            if revised:
                eval_spec.rubric.criteria = revised
                revised_count += 1
                click.echo(f"    Revised criteria:")
                for c in revised:
                    click.echo(f"      - {c}")
            else:
                click.echo("    No revision produced — criteria unchanged.")

        click.echo()
        if revised_count > 0:
            from prompterator.core.eval_spec import save_eval_file
            save_eval_file(eval_file, evals_path)
            click.echo(f"Wrote {revised_count} revised eval(s) to: {evals_path}")
            click.echo("Run 'prompterator calibrate' again to verify the revised criteria.")
        else:
            click.echo("No revisions were produced. Consider editing evals manually.")

    elif needs_fix:
        # Default: suggest a prompterator evals -d command for each bad eval
        click.echo()
        eval_by_id = {ev.id: ev for ev in eval_file.evals}
        for cal in needs_fix:
            eval_spec = eval_by_id.get(cal.eval_id)
            if not eval_spec:
                continue

            def _texts_for(predicate) -> list[str]:
                out: list[str] = []
                for ex in cal.examples:
                    if ex.match or not predicate(ex):
                        continue
                    for fb in feedback_list:
                        mb_name = Path(fb.source_file).name
                        for entry in fb.entries:
                            src = Path(entry.file_ref).name if entry.file_ref else mb_name
                            if src == ex.source and entry.text.strip():
                                out.append(entry.text)
                return out

            missed_texts = _texts_for(
                lambda e: e.eval_result == "PASS" and e.label == "FAIL"
            )
            false_alarm_texts = _texts_for(
                lambda e: e.eval_result == "FAIL" and e.label == "PASS"
            )

            directive_parts = []
            if eval_spec.rubric and eval_spec.rubric.criteria:
                summary = []
                if cal.false_negatives:
                    summary.append(f"missed {cal.false_negatives} known problem(s)")
                if cal.false_positives:
                    summary.append(f"false-alarmed on {cal.false_positives} clean output(s)")
                directive_parts.append(
                    f"The eval {cal.eval_id} " + " and ".join(summary) + ". "
                    f"Current criterion: {eval_spec.rubric.criteria[0]}"
                )
            if missed_texts:
                directive_parts.append(
                    "Missed feedback: " + "; ".join(missed_texts[:3])
                )
            if false_alarm_texts:
                directive_parts.append(
                    "False-alarm feedback (clean outputs wrongly flagged): "
                    + "; ".join(false_alarm_texts[:3])
                )
            if cal.false_negatives and cal.false_positives:
                directive_parts.append("Rebalance the criterion")
            elif cal.false_negatives:
                directive_parts.append("Broaden the criterion to catch these cases")
            else:
                directive_parts.append("Narrow the criterion to avoid these false alarms")
            directive = ". ".join(directive_parts)

            click.echo(f"  {click.style(cal.eval_id, fg='cyan')}: {cal.verdict}")
            click.echo(f"    To fix, re-generate evals with a directive:")
            click.echo()
            click.echo(f"    prompterator evals -d \"{directive}\"")
            click.echo()

        click.echo(
            "Or use --auto-fix to have the LLM revise criteria directly."
        )
        raise SystemExit(1)
