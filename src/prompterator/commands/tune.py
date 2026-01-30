"""Tune command - run the full tuning loop."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.eval_spec import load_eval_file
from prompterator.core.issue import load_issue_file
from prompterator.core.tuner import run_tuning_loop
from prompterator.runners.llm import LLMClient, LLMError


@click.command("tune")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--issues",
    "issues_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to issue file (default: auto-detect from prompt name)",
)
@click.option(
    "--evals",
    "evals_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to eval file (default: auto-detect from prompt name)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of tuning iterations",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for results (default: .prompterator/tune)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without running LLM calls",
)
def tune_cmd(
    prompt: Path,
    issues_path: Path | None,
    evals_path: Path | None,
    max_iterations: int,
    output_dir: Path | None,
    dry_run: bool,
) -> None:
    """Run the full tuning loop: improve → test → improve iteratively.

    PROMPT is the path to the prompt file to tune.
    """
    config = load_config()
    base_dir = get_config_base_dir()
    base_name = prompt.stem.split(".")[0]

    # Find issue file
    if issues_path is None:
        issues_dir = config.get_dir("issues", base_dir)
        issues_path = issues_dir / f"{base_name}.issue.yaml"
        if not issues_path.exists():
            click.echo(f"No issue file found at {issues_path}")
            click.echo("Run 'prompterator issues' first or specify --issues path.")
            raise SystemExit(1)

    # Find eval file
    if evals_path is None:
        evals_dir = config.get_dir("evals", base_dir)
        evals_path = evals_dir / f"{base_name}.eval.yaml"
        if not evals_path.exists():
            click.echo(f"No eval file found at {evals_path}")
            click.echo("Run 'prompterator evals' first or specify --evals path.")
            raise SystemExit(1)

    try:
        issue_file = load_issue_file(issues_path)
        eval_file = load_eval_file(evals_path)
    except Exception as e:
        click.echo(f"Error loading files: {e}", err=True)
        raise SystemExit(1)

    if not issue_file.issues:
        click.echo("No issues to address.")
        raise SystemExit(1)

    if not eval_file.evals:
        click.echo("No evaluations to run.")
        raise SystemExit(1)

    click.echo(f"Tuning: {prompt}")
    click.echo(f"Issues: {len(issue_file.issues)} from {issues_path.name}")
    click.echo(f"Evals: {len(eval_file.evals)} from {evals_path.name}")
    click.echo(f"Max iterations: {max_iterations}")
    click.echo()

    if dry_run:
        click.echo("[dry-run] Would run tuning loop with the above configuration.")
        click.echo(f"[dry-run] Output directory: {output_dir or '.prompterator/tune'}")
        return

    # Set up output directory
    if output_dir is None:
        output_dir = base_dir / ".prompterator" / "tune"

    # Initialize LLM clients
    critic_llm = None
    critic_script = None
    critic_script_timeout = config.critic.script_timeout

    try:
        editor_llm = LLMClient(
            runner=config.editor.runner,
            temperature=config.editor.temperature,
            max_tokens=config.editor.max_tokens,
            model=config.editor.model,
            endpoint=config.editor.endpoint,
            api_version=config.editor.api_version,
        )
        if config.critic.mode == "script":
            critic_script = config.critic.script
            click.echo(f"Critic mode: script ({critic_script})")
        else:
            click.echo("Critic mode: llm")
            critic_llm = LLMClient(
                runner=config.critic.runner,
                temperature=config.critic.temperature,
                max_tokens=config.critic.max_tokens,
                model=config.critic.model,
                endpoint=config.critic.endpoint,
                api_version=config.critic.api_version,
            )
    except LLMError as e:
        click.echo(f"LLM error: {e}", err=True)
        raise SystemExit(1)

    def on_iteration(record):
        verdict_color = {"PASS": "green", "FAIL": "red", "PARTIAL": "yellow"}.get(
            record.summary.verdict, "white"
        )
        click.echo(
            f"  Iteration {record.iteration}: "
            f"score={record.summary.overall_score:.2f} "
            f"[{click.style(record.summary.verdict, fg=verdict_color)}] "
            f"- {record.rationale[:80]}"
        )

    click.echo("Running tuning loop...")
    try:
        report = run_tuning_loop(
            prompt_path=prompt,
            issue_file=issue_file,
            eval_file=eval_file,
            editor_llm=editor_llm,
            critic_llm=critic_llm,
            max_iterations=max_iterations,
            output_dir=output_dir,
            on_iteration=on_iteration,
            critic_script=critic_script,
            critic_script_timeout=critic_script_timeout,
        )
    except LLMError as e:
        click.echo(f"LLM error during tuning: {e}", err=True)
        raise SystemExit(1)

    # Display final results
    click.echo()
    click.echo("=" * 50)
    click.echo("Tuning Complete")
    click.echo("=" * 50)
    click.echo(f"Iterations: {len(report.iterations)}")

    verdict_color = {"PASS": "green", "FAIL": "red", "PARTIAL": "yellow"}.get(
        report.final_summary.verdict, "white"
    )
    click.echo(
        f"Final verdict: {click.style(report.final_summary.verdict, fg=verdict_color, bold=True)}"
    )
    click.echo(f"Final score: {report.final_summary.overall_score:.2f}")

    if report.metric_table:
        click.echo()
        click.echo("Metric Table:")
        click.echo(f"  {'Eval ID':<30} {'Before':>8} {'After':>8} {'Delta':>8}")
        click.echo(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for row in report.metric_table:
            delta_str = f"{row['delta']:+.4f}"
            click.echo(
                f"  {row['eval_id']:<30} {row['before']:>8.4f} {row['after']:>8.4f} {delta_str:>8}"
            )

    click.echo(f"\nResults saved to: {output_dir}")
