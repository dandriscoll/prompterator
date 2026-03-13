"""Calibrate command - verify evals agree with human-labeled feedback."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.calibrator import calibrate, save_calibration_report
from prompterator.core.run import create_run_dir
from prompterator.commands.resolve import ResolveError, resolve_prompt_and_evals, resolve_issues
from prompterator.commands.feedback import find_mb_files, parse_mb_file
from prompterator.models.calibration import CalibrationReport
from prompterator.runners.llm import LLMClient, LLMError


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
def calibrate_cmd(
    prompt: Path | None,
    evals_path: Path | None,
    feedback_dir: Path | None,
    output: Path | None,
    verbose: bool,
) -> None:
    """Verify that evals agree with human-labeled feedback.

    PROMPT is the path to the prompt file (optional — can be derived
    from eval/issue files).

    Runs each eval against the labeled feedback examples and reports
    whether the eval verdicts match the human labels. This validates
    that evals are reliable before using them to drive the tuning loop.
    """
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
    click.echo(f"Calibrating: {prompt_ref}")
    click.echo(f"Evals: {len(eval_file.evals)} from {evals_path.name}")
    click.echo(f"Feedback files: {len(feedback_list)}")
    click.echo()

    # --- Initialize LLM ---------------------------------------------------
    try:
        llm = LLMClient(**config.resolve_role("critic"))
    except LLMError as e:
        click.echo(f"Critic LLM error: {e}", err=True)
        raise SystemExit(1)

    # --- Run calibration --------------------------------------------------
    from prompterator.runners.llm import debug_context
    debug_context("calibrate")
    click.echo("Running calibration...")
    try:
        cal_results = calibrate(eval_file, feedback_list, issue_file, llm)
    except LLMError as e:
        click.echo(f"LLM error during calibration: {e}", err=True)
        raise SystemExit(1)

    if not cal_results:
        click.echo("No calibration results produced.")
        raise SystemExit(1)

    # --- Display results --------------------------------------------------
    any_bad = False

    for cal in cal_results:
        click.echo()
        click.echo(f"Eval: {cal.eval_id}")

        if verbose:
            click.echo()
            header = f"{'Example':<35} {'Label':<10} {'Eval':<10} {'Match'}"
            click.echo(header)
            click.echo("-" * len(header))
            for ex in cal.examples:
                match_str = "yes" if ex.match else click.style("NO", fg="red", bold=True)
                click.echo(f"{ex.source:<35} {ex.label:<10} {ex.eval_result:<10} {match_str}")
            click.echo("-" * len(header))

        pct = cal.accuracy * 100
        click.echo(
            f"Accuracy: {int(cal.accuracy * cal.num_examples)}/{cal.num_examples} ({pct:.1f}%)"
        )
        click.echo(f"Precision: {cal.precision:.2f}  Recall: {cal.recall:.2f}  F1: {cal.f1:.2f}")
        click.echo(f"False positives: {cal.false_positives}  False negatives: {cal.false_negatives}")

        verdict_color = {"GOOD": "green", "WEAK": "yellow", "BAD": "red"}[cal.verdict]
        reason = {
            "GOOD": "accuracy >= 80%",
            "WEAK": "accuracy >= 60%",
            "BAD": "accuracy < 60%",
        }[cal.verdict]
        click.echo(
            f"Verdict: {click.style(cal.verdict, fg=verdict_color, bold=True)} ({reason})"
        )

        if cal.verdict == "BAD":
            any_bad = True

    # --- Save report ------------------------------------------------------
    base_name = prompt.stem.split(".")[0]
    if output is None:
        results_dir = config.get_dir("results", base_dir)
        run_dir = create_run_dir(results_dir)
        output = run_dir / f"{base_name}.calibration.yaml"

    report = CalibrationReport(
        prompt_ref=prompt_ref,
        eval_file=str(evals_path),
        calibrations=cal_results,
    )
    save_calibration_report(report, output)
    click.echo(f"\nCalibration report saved to: {output}")

    if any_bad:
        click.echo(
            "\nOne or more evals have BAD calibration. "
            "Consider revising them before running improve/tune."
        )
        raise SystemExit(1)
