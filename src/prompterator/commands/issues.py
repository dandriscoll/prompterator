"""Issues command - consolidate feedback into .issue.yaml files."""

from collections import defaultdict
from pathlib import Path

import click

from prompterator.commands.feedback import find_mb_files, parse_mb_file
from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.issue import _split_feedback_entry, consolidate_feedback, load_issue_file, save_issue_file
from prompterator.runners.llm import LLMClient


@click.command("issues")
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for .mb files (default: from config)",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for issue files (default: from config)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be created without writing files (still calls LLM for clustering)",
)
@click.option(
    "--directive",
    "-d",
    type=str,
    default=None,
    help="Guidance for the LLM when clustering feedback into issues",
)
def issues_cmd(directory: Path | None, output: Path | None, dry_run: bool, directive: str | None) -> None:
    """Consolidate feedback into .issue.yaml files."""
    config = load_config()
    base_dir = get_config_base_dir()

    if directory is None:
        directory = config.get_dir("feedback", base_dir)

    if output is None:
        output = config.get_dir("issues", base_dir)

    mb_files = find_mb_files(directory)

    if not mb_files:
        click.echo(f"No .mb files found in {directory}")
        return

    # Initialize LLM client with editor role
    from prompterator.runners.llm import debug_context
    debug_context("issues")
    llm_client = LLMClient(**config.resolve_role("editor"))

    # Resolve primary prompt from config if set
    config_prompt = config.directories.prompt

    # Group feedback by prompt reference
    prompt_feedback: dict[str, list] = defaultdict(list)

    for path in mb_files:
        try:
            feedback = parse_mb_file(path)
            if config_prompt:
                # Config specifies the primary prompt — all feedback maps to it
                prompt_feedback[config_prompt].append(feedback)
            elif feedback.prompt_ref:
                prompt_feedback[feedback.prompt_ref].append(feedback)
            else:
                # Use filename as key if no prompt ref
                stem = path.stem
                prompt_feedback[f"{stem}.prompt.txt"].append(feedback)
        except Exception as e:
            click.echo(f"Error parsing {path}: {e}", err=True)

    if not prompt_feedback:
        click.echo("No feedback to consolidate")
        return

    # Generate issue files
    created = 0
    for prompt_ref, feedback_list in prompt_feedback.items():
        # Load existing issues if present
        base_name = Path(prompt_ref).stem.split(".")[0]
        issue_path = output / f"{base_name}.issue.yaml"

        existing_issues = None
        if issue_path.exists():
            try:
                existing = load_issue_file(issue_path)
                existing_issues = existing.issues
                click.echo(f"  {prompt_ref}: merging with {len(existing_issues)} existing issues")
            except Exception as e:
                click.echo(f"  Warning: could not load {issue_path}: {e}", err=True)

        issue_file = consolidate_feedback(
            feedback_list,
            prompt_ref,
            llm_client,
            config.feedback.min_occurrences,
            existing_issues=existing_issues,
            directive=directive,
        )

        if not issue_file.issues:
            n_obs = sum(
                len(_split_feedback_entry(e.text))
                for fb in feedback_list for e in fb.entries
            )
            click.echo(
                f"  {prompt_ref}: no issues generated from {len(feedback_list)} file(s), "
                f"{n_obs} observation(s) — check min_occurrences ({config.feedback.min_occurrences}) "
                f"or re-run with --dry-run to see LLM output"
            )
            continue

        if dry_run:
            click.echo(f"  Would create: {issue_path}")
            click.echo(f"    Issues: {len(issue_file.issues)}")
            for issue in issue_file.issues:
                click.echo(f"      - [{issue.severity}] {issue.category}: {issue.summary}")
        else:
            save_issue_file(issue_file, issue_path)
            click.echo(f"  Created: {issue_path} ({len(issue_file.issues)} issues)")
            created += 1

    if not dry_run:
        click.echo(f"\nCreated {created} issue file(s) in {output}")
