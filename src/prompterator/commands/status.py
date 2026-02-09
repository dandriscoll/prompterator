"""Status command - show workflow state."""

from pathlib import Path

import click

from prompterator.config.loader import (
    CONFIG_FILENAME,
    find_config_file,
    get_config_base_dir,
    load_config,
)


def count_files(directory: Path, pattern: str) -> int:
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


@click.command("status")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed file listings",
)
def status_cmd(verbose: bool) -> None:
    """Show the current workflow state."""
    config_path = find_config_file()

    if config_path is None:
        click.echo("No prompterator.yaml found.")
        click.echo("Run 'prompterator init' to create one.")
        return

    config = load_config(config_path)
    base_dir = get_config_base_dir(config_path)

    click.echo(f"Config: {config_path}")
    click.echo()

    # Get directories
    prompts_dir = config.get_dir("prompts", base_dir)
    feedback_dir = config.get_dir("feedback", base_dir)
    issues_dir = config.get_dir("issues", base_dir)
    evals_dir = config.get_dir("evals", base_dir)
    results_dir = config.get_dir("results", base_dir)

    # Count files
    prompt_count = count_files(prompts_dir, "*.prompt.txt") + count_files(
        prompts_dir, "*.prompt.md"
    )
    feedback_count = count_files(feedback_dir, "*.mb")
    issue_count = count_files(issues_dir, "*.issue.yaml")
    eval_count = count_files(evals_dir, "*.eval.yaml")
    result_count = count_files(results_dir, "*.results.yaml")

    # Display status
    click.echo("Workflow Status:")
    click.echo("-" * 40)

    # Prompts
    status_icon = "+" if prompt_count > 0 else "-"
    click.echo(f"  [{status_icon}] Prompts:  {prompt_count:3d}  ({prompts_dir})")
    if verbose and prompt_count > 0:
        for f in sorted(prompts_dir.glob("*.prompt.*"))[:10]:
            click.echo(f"        {f.name}")

    # Feedback
    status_icon = "+" if feedback_count > 0 else "-"
    click.echo(f"  [{status_icon}] Feedback: {feedback_count:3d}  ({feedback_dir})")
    if verbose and feedback_count > 0:
        for f in sorted(feedback_dir.glob("*.mb"))[:10]:
            click.echo(f"        {f.name}")

    # Issues
    status_icon = "+" if issue_count > 0 else "-"
    click.echo(f"  [{status_icon}] Issues:   {issue_count:3d}  ({issues_dir})")
    if verbose and issue_count > 0:
        for f in sorted(issues_dir.glob("*.issue.yaml"))[:10]:
            click.echo(f"        {f.name}")

    # Evals
    status_icon = "+" if eval_count > 0 else "-"
    click.echo(f"  [{status_icon}] Evals:    {eval_count:3d}  ({evals_dir})")
    if verbose and eval_count > 0:
        for f in sorted(evals_dir.glob("*.eval.yaml"))[:10]:
            click.echo(f"        {f.name}")

    # Results
    status_icon = "+" if result_count > 0 else "-"
    click.echo(f"  [{status_icon}] Results:  {result_count:3d}  ({results_dir})")
    if verbose and result_count > 0:
        for f in sorted(results_dir.glob("*.results.yaml"))[:10]:
            click.echo(f"        {f.name}")

    click.echo("-" * 40)

    # Suggest next steps
    click.echo()
    click.echo("Next steps:")
    if prompt_count == 0:
        click.echo("  - Create prompt files (*.prompt.txt or *.prompt.md)")
    elif feedback_count == 0:
        click.echo("  - Add feedback in .mb files for your prompts")
    elif issue_count == 0:
        click.echo("  - Run 'prompterator issues' to consolidate feedback")
    elif eval_count == 0:
        click.echo("  - Run 'prompterator evals' to generate evaluations")
    elif result_count == 0:
        click.echo("  - Run 'prompterator improve <prompt>' to generate improvements")
        click.echo("  - Run 'prompterator test <prompt>' to run evaluations")
    else:
        click.echo("  - Workflow complete! Review results in " + str(results_dir))

    # LLM Role Configurations
    click.echo()
    click.echo("LLM Roles:")
    for role_name in ("author", "editor", "critic"):
        resolved = config.resolve_role(role_name)
        role = getattr(config, role_name)
        click.echo(f"  {role_name.title()}: stack={role.stack}, runner={resolved['runner']} (temp={resolved['temperature']})")
