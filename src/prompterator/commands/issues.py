"""Issues command - consolidate feedback into .issue.yaml files."""

from collections import defaultdict
from pathlib import Path

import click

from prompterator.commands.feedback import find_mb_files, parse_mb_file
from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.issue import consolidate_feedback, save_issue_file


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
    help="Show what would be created without writing files",
)
def issues_cmd(directory: Path | None, output: Path | None, dry_run: bool) -> None:
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

    # Group feedback by prompt reference
    prompt_feedback: dict[str, list] = defaultdict(list)

    for path in mb_files:
        try:
            feedback = parse_mb_file(path)
            if feedback.prompt_ref:
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
        issue_file = consolidate_feedback(
            feedback_list,
            prompt_ref,
            config.feedback.categories,
            config.feedback.min_occurrences,
        )

        if not issue_file.issues:
            click.echo(f"  {prompt_ref}: no issues (below threshold)")
            continue

        # Generate output filename
        base_name = Path(prompt_ref).stem.split(".")[0]
        issue_path = output / f"{base_name}.issue.yaml"

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
