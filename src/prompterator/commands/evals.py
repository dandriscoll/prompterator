"""Evals command - generate .eval.yaml files from issues."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.eval_spec import generate_evals_from_issues, load_eval_file, save_eval_file
from prompterator.core.issue import load_issue_file
from prompterator.runners.llm import LLMClient


@click.command("evals")
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for .issue.yaml files (default: from config)",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for eval files (default: from config)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be created without writing files",
)
@click.option(
    "--directive",
    "-d",
    type=str,
    default=None,
    help="Guidance for the LLM when generating eval criteria from issues",
)
def evals_cmd(directory: Path | None, output: Path | None, dry_run: bool, directive: str | None) -> None:
    """Generate .eval.yaml files from issues."""
    config = load_config()
    base_dir = get_config_base_dir()

    if directive is None:
        directive = config.resolve_directive("evals")

    if directory is None:
        directory = config.get_dir("issues", base_dir)

    if output is None:
        output = config.get_dir("evals", base_dir)

    # Find issue files
    issue_files = sorted(directory.glob("*.issue.yaml"))

    if not issue_files:
        click.echo(f"No .issue.yaml files found in {directory}")
        click.echo("Run 'prompterator issues' first to generate issue files.")
        return

    # Initialize LLM client for criteria inversion
    from prompterator.runners.llm import debug_context
    debug_context("evals")
    llm_client = LLMClient(**config.resolve_role("editor"))

    created = 0
    for issue_path in issue_files:
        try:
            issue_file = load_issue_file(issue_path)
        except Exception as e:
            click.echo(f"Error loading {issue_path}: {e}", err=True)
            continue

        if not issue_file.issues:
            click.echo(f"  {issue_path.name}: no issues to convert")
            continue

        # Load existing evals if present
        base_name = issue_path.stem.replace(".issue", "")
        eval_path = output / f"{base_name}.eval.yaml"

        existing_evals = None
        if eval_path.exists():
            try:
                existing = load_eval_file(eval_path)
                existing_evals = existing.evals
                # Check for issue reorganization
                new_ids = {issue.id for issue in issue_file.issues}
                orphaned = [ev for ev in existing_evals if ev.issue_ref and ev.issue_ref not in new_ids]
                if orphaned:
                    click.echo(
                        f"  {issue_path.name}: issues reorganized — reconciling "
                        f"{len(existing_evals)} existing evals with {len(issue_file.issues)} new issues"
                    )
                else:
                    click.echo(f"  {issue_path.name}: merging with {len(existing_evals)} existing evals")
            except Exception as e:
                click.echo(f"  Warning: could not load {eval_path}: {e}", err=True)

        eval_file = generate_evals_from_issues(
            issue_file, llm_client, existing_evals=existing_evals,
            directive=directive,
        )

        if dry_run:
            click.echo(f"  Would create: {eval_path}")
            click.echo(f"    Evals: {len(eval_file.evals)}")
            for ev in eval_file.evals:
                click.echo(f"      - {ev.id}: {ev.description}")
        else:
            save_eval_file(eval_file, eval_path)
            click.echo(f"  Created: {eval_path} ({len(eval_file.evals)} evals)")
            created += 1

    if not dry_run:
        click.echo(f"\nCreated {created} eval file(s) in {output}")
