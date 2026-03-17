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


def _heading(text: str) -> str:
    return click.style(text, bold=True)


def _dim(text: str) -> str:
    return click.style(text, dim=True)


def _severity_color(severity: str) -> str:
    colors = {"high": "red", "medium": "yellow", "low": "green"}
    return click.style(severity, fg=colors.get(severity, "white"))


def _wrap(text: str, indent: int = 6, width: int = 78) -> str:
    """Word-wrap text with hanging indent."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = " " * indent + word
        else:
            current = current + " " + word if current else word
    if current:
        lines.append(current)
    return "\n".join(lines)


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

    click.echo()
    if prompt:
        click.echo(f"{_heading('Prompt')}  {prompt}")
    else:
        click.echo(f"{_heading('Prompt')}  {_dim('(not configured)')}")

    # --- Content ----------------------------------------------------------
    content_pairs = resolve_content_with_paths(config, base_dir)
    click.echo()
    if content_pairs:
        click.echo(f"{_heading('Content')}  {len(content_pairs)} file(s)")
        for p, text in content_pairs:
            lines = len(text.splitlines())
            click.echo(f"  {click.style(p.name, fg='cyan')}  {_dim(f'{lines} lines')}")
    else:
        click.echo(f"{_heading('Content')}  {_dim('(none)')}")

    # --- Counterpart ------------------------------------------------------
    counterpart_dirs = resolve_counterpart(config, base_dir)
    if counterpart_dirs:
        click.echo()
        click.echo(f"{_heading('Counterpart')}  {len(counterpart_dirs)} directions file(s)")

    # --- Feedback ---------------------------------------------------------
    feedback_dir = config.get_dir("feedback", base_dir)
    click.echo()
    if feedback_dir.exists():
        mb_files = find_mb_files(feedback_dir)
        if mb_files:
            all_feedback = []
            for path in mb_files:
                try:
                    all_feedback.append((path, parse_mb_file(path)))
                except Exception:
                    all_feedback.append((path, None))

            total_entries = sum(len(fb.entries) for _, fb in all_feedback if fb)
            click.echo(
                f"{_heading('Feedback')}  {len(mb_files)} file(s), "
                f"{total_entries} entries"
            )
            for path, fb in all_feedback:
                if fb:
                    click.echo(f"  {click.style(path.name, fg='cyan')}  {_dim(f'{len(fb.entries)} entries')}")
                else:
                    click.echo(f"  {click.style(path.name, fg='cyan')}  {click.style('parse error', fg='red')}")
        else:
            click.echo(f"{_heading('Feedback')}  {_dim(f'(no .mb files in {feedback_dir})')}")
    else:
        click.echo(f"{_heading('Feedback')}  {_dim('(directory not found)')}")

    # --- Issues -----------------------------------------------------------
    click.echo()
    if prompt:
        try:
            issues_path, issue_file = resolve_issues(config, base_dir, prompt)
            click.echo(f"{_heading('Issues')}  {len(issue_file.issues)} from {issues_path.name}")
            for issue in issue_file.issues:
                evidence_count = len(issue.evidence)
                sev = _severity_color(issue.severity)
                click.echo(
                    f"  {click.style(issue.id, fg='cyan')}  "
                    f"[{sev}] {_dim(issue.category)}"
                )
                click.echo(f"      {_wrap(issue.summary, indent=6)}")
                sources = sorted({ev.source for ev in issue.evidence})
                click.echo(f"      {_dim(f'{evidence_count} evidence from: {", ".join(sources)}')}")
        except ResolveError:
            click.echo(f"{_heading('Issues')}  {_dim('(not generated yet)')}")
    else:
        click.echo(f"{_heading('Issues')}  {_dim('(no prompt to resolve from)')}")

    # --- Evals ------------------------------------------------------------
    click.echo()
    if prompt:
        try:
            _, evals_path, eval_file = resolve_prompt_and_evals(
                config, base_dir, prompt, None,
            )
            click.echo(f"{_heading('Evals')}  {len(eval_file.evals)} from {evals_path.name}")
            for ev in eval_file.evals:
                click.echo(f"  {click.style(ev.id, fg='cyan')}")
                if ev.description:
                    click.echo(f"      {_wrap(ev.description, indent=6)}")
                if ev.issue_ref:
                    click.echo(f"      {_dim('Issue:')} {ev.issue_ref}")
                if ev.rubric and ev.rubric.criteria:
                    for c in ev.rubric.criteria:
                        click.echo(f"      {click.style('Criterion:', fg='green')} {_wrap(c, indent=17)}")
                if ev.assertion:
                    click.echo(f"      {click.style('Assertion:', fg='green')} {_wrap(ev.assertion, indent=17)}")
        except ResolveError:
            click.echo(f"{_heading('Evals')}  {_dim('(not generated yet)')}")
    else:
        click.echo(f"{_heading('Evals')}  {_dim('(no prompt to resolve from)')}")

    click.echo()
