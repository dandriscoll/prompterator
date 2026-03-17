"""Summarize command - print a summary of all inputs, issues, and evals."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.commands.feedback import find_mb_files, parse_mb_file
from prompterator.commands.resolve import (
    ResolveError,
    resolve_content_with_paths,
    resolve_counterpart,
    resolve_prompt,
    resolve_prompt_and_evals,
    resolve_issues,
)


@click.command("summarize")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
def summarize_cmd(prompt: Path | None) -> None:
    """Print a summary of feedback, content, issues, and evals.

    Shows the current state of all workflow artifacts for a prompt:
    feedback files and entry counts, content files, issue IDs with
    summaries, and eval IDs with criteria.
    """
    config = load_config()
    base_dir = get_config_base_dir()

    # --- Prompt -----------------------------------------------------------
    if prompt is None:
        prompt = resolve_prompt(config, base_dir)
    if prompt:
        click.echo(f"Prompt: {prompt}")
    else:
        click.echo("Prompt: (not configured)")

    # --- Content ----------------------------------------------------------
    content_pairs = resolve_content_with_paths(config, base_dir)
    if content_pairs:
        click.echo(f"\nContent: {len(content_pairs)} file(s)")
        for p, text in content_pairs:
            lines = len(text.splitlines())
            click.echo(f"  {p.name} ({lines} lines)")
    else:
        click.echo("\nContent: (none)")

    # --- Counterpart ------------------------------------------------------
    counterpart_dirs = resolve_counterpart(config, base_dir)
    if counterpart_dirs:
        click.echo(f"\nCounterpart: {len(counterpart_dirs)} directions file(s)")

    # --- Feedback ---------------------------------------------------------
    feedback_dir = config.get_dir("feedback", base_dir)
    if feedback_dir.exists():
        mb_files = find_mb_files(feedback_dir)
        if mb_files:
            total_entries = 0
            for path in mb_files:
                try:
                    fb = parse_mb_file(path)
                    total_entries += len(fb.entries)
                except Exception:
                    pass
            click.echo(f"\nFeedback: {len(mb_files)} .mb file(s), {total_entries} entries")
            for path in mb_files:
                try:
                    fb = parse_mb_file(path)
                    click.echo(f"  {path.name} ({len(fb.entries)} entries)")
                except Exception:
                    click.echo(f"  {path.name} (parse error)")
        else:
            click.echo(f"\nFeedback: (no .mb files in {feedback_dir})")
    else:
        click.echo("\nFeedback: (directory not found)")

    # --- Issues -----------------------------------------------------------
    if prompt:
        try:
            issues_path, issue_file = resolve_issues(config, base_dir, prompt)
            click.echo(f"\nIssues: {len(issue_file.issues)} from {issues_path.name}")
            for issue in issue_file.issues:
                evidence_count = len(issue.evidence)
                click.echo(
                    f"  {issue.id} [{issue.severity}] ({issue.category})"
                )
                click.echo(f"    {issue.summary}")
                click.echo(f"    Evidence: {evidence_count} source(s)")
        except ResolveError:
            click.echo("\nIssues: (not generated yet)")
    else:
        click.echo("\nIssues: (no prompt to resolve from)")

    # --- Evals ------------------------------------------------------------
    if prompt:
        try:
            _, evals_path, eval_file = resolve_prompt_and_evals(
                config, base_dir, prompt, None,
            )
            click.echo(f"\nEvals: {len(eval_file.evals)} from {evals_path.name}")
            for ev in eval_file.evals:
                click.echo(f"  {ev.id}")
                if ev.description:
                    click.echo(f"    {ev.description}")
                if ev.issue_ref:
                    click.echo(f"    Issue: {ev.issue_ref}")
                if ev.rubric and ev.rubric.criteria:
                    for c in ev.rubric.criteria:
                        click.echo(f"    Criterion: {c}")
                if ev.assertion:
                    click.echo(f"    Assertion: {ev.assertion}")
        except ResolveError:
            click.echo("\nEvals: (not generated yet)")
    else:
        click.echo("\nEvals: (no prompt to resolve from)")
