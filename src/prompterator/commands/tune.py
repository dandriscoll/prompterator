"""Tune command - run the full tuning loop."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.commands.resolve import (
    ResolveError,
    resolve_content_with_paths,
    resolve_counterpart,
    resolve_feedback,
    resolve_issues,
    resolve_prompt_and_evals,
)
from prompterator.core.eval_runner import map_content_to_evals
from prompterator.core.tuner import run_tuning_loop
from prompterator.runners.llm import LLMClient, LLMError


@click.command("tune")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
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
    "--runs",
    "-r",
    type=int,
    default=None,
    help="Exact number of iterations to run (no early stopping)",
)
@click.option(
    "--max-runs",
    "-m",
    type=int,
    default=None,
    help="Maximum iterations with early stopping on plateau (default: 10)",
)
@click.option(
    "--samples",
    "-s",
    type=int,
    default=None,
    help="Author outputs per content file (default: from config)",
)
@click.option(
    "--ensemble",
    "-e",
    type=int,
    default=None,
    help="Critic evaluations per output (default: from config critic.ensemble)",
)
@click.option(
    "--patience",
    "-p",
    type=int,
    default=5,
    show_default=True,
    help="Early stop after N non-improving iterations (only with -m)",
)
@click.option(
    "--content",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Content file to pair with the prompt (overrides config)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without running LLM calls",
)
@click.option(
    "--all-evals",
    is_flag=True,
    help="Run all evals for every content file (skip feedback-based filtering)",
)
@click.option(
    "--counterpart",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Counterpart directions file for multi-turn dialog (overrides config)",
)
@click.option(
    "--dialog-turns",
    type=int,
    default=3,
    show_default=True,
    help="Number of author turns in multi-turn dialog",
)
@click.option(
    "--focus",
    "-f",
    type=str,
    default=None,
    help="Focus on improving a specific eval ID (others must not regress)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress live progress output",
)
def tune_cmd(
    prompt: Path | None,
    issues_path: Path | None,
    evals_path: Path | None,
    runs: int | None,
    max_runs: int | None,
    samples: int | None,
    ensemble: int | None,
    patience: int,
    content: Path | None,
    dry_run: bool,
    all_evals: bool,
    counterpart: Path | None,
    dialog_turns: int,
    focus: str | None,
    quiet: bool,
) -> None:
    """Run the full tuning loop: improve → test → improve iteratively.

    PROMPT is the path to the prompt file to tune (optional — can be
    derived from issue/eval files).
    """
    config = load_config()
    base_dir = get_config_base_dir()

    try:
        prompt, evals_path, eval_file = resolve_prompt_and_evals(
            config, base_dir, prompt, evals_path,
        )
        issues_path, issue_file = resolve_issues(
            config, base_dir, prompt, issues_path,
        )
    except ResolveError as e:
        click.echo(str(e))
        raise SystemExit(1)

    if not issue_file.issues:
        click.echo("No issues to address.")
        raise SystemExit(1)

    if not eval_file.evals:
        click.echo("No evaluations to run.")
        raise SystemExit(1)

    if focus:
        eval_ids = {ev.id for ev in eval_file.evals}
        if focus not in eval_ids:
            click.echo(f"Focus eval '{focus}' not found. Available: {', '.join(sorted(eval_ids))}")
            raise SystemExit(1)
        click.echo(f"Focus: {focus} (others must not regress)")

    if runs is not None and max_runs is not None:
        click.echo("Error: --runs/-r and --max-runs/-m are mutually exclusive.", err=True)
        raise SystemExit(1)

    if runs is not None:
        # Exact mode: run exactly N iterations, no early stopping
        n_iterations = runs
        effective_patience = runs  # effectively disabled
        early_stop = False
    else:
        # Max mode (default): run up to N iterations with early stopping
        n_iterations = max_runs if max_runs is not None else 10
        effective_patience = patience
        early_stop = True

    n_evals = len(eval_file.evals)
    n_samples = samples if samples is not None else config.critic.samples
    n_ensemble = ensemble if ensemble is not None else config.critic.ensemble

    # Resolve content files (with paths for feedback mapping)
    content_pairs = resolve_content_with_paths(config, base_dir, content)
    if content_pairs:
        content_paths = [p for p, _ in content_pairs]
        content_texts = [t for _, t in content_pairs]
    else:
        content_paths = []
        content_texts = [None]
    n_content = len(content_texts)

    # --- Feedback-based eval filtering ------------------------------------
    content_eval_map: dict[int, list[str]] | None = None

    if not all_evals and content_paths:
        feedback_list = resolve_feedback(
            config, base_dir, eval_file.prompt_ref,
        )
        if feedback_list:
            content_eval_map = map_content_to_evals(
                content_paths, feedback_list, issue_file, eval_file,
            )

    if content_eval_map is not None:
        total_pairs = sum(len(eids) for eids in content_eval_map.values())
        full_pairs = n_content * n_evals
        click.echo(
            f"Feedback filtering: {total_pairs}/{full_pairs} content-eval pairs "
            f"(skipping evals without feedback on each content)"
        )
        click.echo("  Use --all-evals to override and run every eval for every content file.")

    # Resolve counterpart directions
    counterpart_directions = resolve_counterpart(config, base_dir, counterpart)

    # LLM calls per test: outputs * (1 author + n_evals * ensemble critic)
    n_outputs = n_content * n_samples
    calls_per_test = n_outputs * (1 + n_evals * n_ensemble)
    # Per iteration: 1 improve + 1 review + 1 test
    calls_per_iter = 2 + calls_per_test
    # Total: baseline test + iterations * per-iteration
    total_llm_calls = calls_per_test + n_iterations * calls_per_iter

    click.echo(f"Tuning: {prompt}")
    click.echo(f"Issues: {len(issue_file.issues)} from {issues_path.name}")
    click.echo(f"Evals: {n_evals} from {evals_path.name}")
    if n_content > 1 or content_texts[0] is not None:
        click.echo(f"Content files: {n_content}")
    if early_stop:
        click.echo(f"Max iterations: {n_iterations}, patience: {effective_patience}")
    else:
        click.echo(f"Iterations: {n_iterations} (exact)")
    click.echo(f"Samples: {n_samples}, ensemble: {n_ensemble}, threshold: {config.critic.confidence_threshold:.1f}/10")
    click.echo(f"LLM calls: {'up to ' if early_stop else ''}{total_llm_calls} ({calls_per_test} baseline + {n_iterations} x {calls_per_iter} per iteration)")
    click.echo()

    results_dir = config.get_dir("results", base_dir)

    if dry_run:
        click.echo("[dry-run] Would run tuning loop with the above configuration.")
        click.echo(f"[dry-run] Results directory: {results_dir}")
        return

    # Initialize LLM clients
    critic_llm = None
    critic_script = None
    critic_script_timeout = config.critic.script_timeout
    counterpart_llm_client = None

    try:
        author_llm = LLMClient(**config.resolve_role("author"))
        editor_llm = LLMClient(**config.resolve_role("editor"))
        if counterpart_directions:
            counterpart_llm_client = LLMClient(**config.resolve_role("counterpart"))
            click.echo(f"Counterpart: {len(counterpart_directions)} directions, {dialog_turns} turns")
        if config.critic.mode == "script":
            critic_script = config.critic.script
            click.echo(f"Critic mode: script ({critic_script})")
        else:
            click.echo("Critic mode: llm")
            critic_llm = LLMClient(**config.resolve_role("critic"))
    except LLMError as e:
        click.echo(f"LLM error: {e}", err=True)
        raise SystemExit(1)

    import sys

    _last_status_len = 0

    def on_status(msg: str):
        nonlocal _last_status_len
        # Overwrite the current line with the status message
        padded = msg[:100].ljust(_last_status_len)
        sys.stderr.write(f"\r  {padded}")
        sys.stderr.flush()
        _last_status_len = len(msg[:100])

    def _clear_status():
        nonlocal _last_status_len
        if _last_status_len:
            sys.stderr.write(f"\r{' ' * (_last_status_len + 2)}\r")
            sys.stderr.flush()
            _last_status_len = 0

    def on_iteration(record):
        _clear_status()
        verdict_color = {"PASS": "green", "FAIL": "red", "PARTIAL": "yellow"}.get(
            record.summary.verdict, "white"
        )
        click.echo(
            f"  Iteration {record.iteration}: "
            f"score={record.summary.overall_score:.1f}/10 "
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
            max_iterations=n_iterations,
            on_iteration=on_iteration,
            on_status=on_status,
            author_llm=author_llm,
            content_texts=content_texts if content_texts != [None] else None,
            samples=n_samples,
            ensemble=n_ensemble,
            confidence_threshold=config.critic.confidence_threshold,
            critic_script=critic_script,
            critic_script_timeout=critic_script_timeout,
            patience=effective_patience,
            early_stop=early_stop,
            results_dir=results_dir,
            content_eval_map=content_eval_map,
            counterpart_llm=counterpart_llm_client,
            counterpart_directions=counterpart_directions or None,
            dialog_turns=dialog_turns,
            quiet=quiet,
            focus_eval=focus,
        )
    except LLMError as e:
        _clear_status()
        click.echo(f"LLM error during tuning: {e}", err=True)
        raise SystemExit(1)

    _clear_status()
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
    click.echo(f"Final score: {report.final_summary.overall_score:.1f}/10")

    if report.metric_table:
        click.echo()
        click.echo("Metric Table:")
        click.echo(f"  {'Eval ID':<30} {'Before':>8} {'After':>8} {'Delta':>8}")
        click.echo(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for row in report.metric_table:
            delta_str = f"{row['delta']:+.1f}"
            click.echo(
                f"  {row['eval_id']:<30} {row['before']:>6.1f}/10 {row['after']:>6.1f}/10 {delta_str:>6}"
            )

    if report.help_request:
        click.echo()
        click.echo("=" * 50)
        click.echo("Help Requested")
        click.echo("=" * 50)
        click.echo(report.help_request)
        click.echo()
        click.echo(
            "Tip: use 'prompterator improve -d \"...\"' to give the editor LLM a specific instruction, then re-run tune."
        )

    click.echo(f"\nPrompt updated: {prompt}")
    click.echo(f"Results saved to: {results_dir}")
