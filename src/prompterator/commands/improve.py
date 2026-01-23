"""Improve command - generate improved prompts via LLM."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.improver import generate_improved_prompt, save_improved_prompt
from prompterator.core.issue import load_issue_file
from prompterator.runners.ft import FTClient, FTError
from prompterator.runners.llm import LLMClient, LLMError


@click.command("improve")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--issues",
    "issues_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to issue file (default: auto-detect from prompt name)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for improved prompt (default: auto-generated)",
)
@click.option(
    "--in-place",
    is_flag=True,
    help="Overwrite the original prompt file (git mode)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the improved prompt without saving",
)
def improve_cmd(
    prompt: Path,
    issues_path: Path | None,
    output: Path | None,
    in_place: bool,
    dry_run: bool,
) -> None:
    """Generate an improved prompt based on identified issues.

    PROMPT is the path to the prompt file to improve.
    """
    config = load_config()
    base_dir = get_config_base_dir()

    # Find issue file if not specified
    if issues_path is None:
        issues_dir = config.get_dir("issues", base_dir)
        base_name = prompt.stem.split(".")[0]
        issues_path = issues_dir / f"{base_name}.issue.yaml"

        if not issues_path.exists():
            click.echo(f"No issue file found at {issues_path}")
            click.echo("Run 'prompterator issues' first or specify --issues path.")
            raise SystemExit(1)

    try:
        issue_file = load_issue_file(issues_path)
    except Exception as e:
        click.echo(f"Error loading issue file: {e}", err=True)
        raise SystemExit(1)

    if not issue_file.issues:
        click.echo("No issues to address in issue file.")
        raise SystemExit(1)

    click.echo(f"Improving: {prompt}")
    click.echo(f"Based on: {len(issue_file.issues)} issues from {issues_path.name}")
    click.echo()

    # Initialize LLM client
    try:
        llm = LLMClient(
            runner=config.llm.runner,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    except LLMError as e:
        click.echo(f"LLM error: {e}", err=True)
        raise SystemExit(1)

    # Generate improved prompt
    click.echo("Generating improved prompt...")
    try:
        improved = generate_improved_prompt(prompt, issue_file, llm)
    except LLMError as e:
        click.echo(f"LLM error: {e}", err=True)
        raise SystemExit(1)

    if dry_run:
        click.echo("\n--- Improved Prompt ---")
        click.echo(improved)
        click.echo("--- End ---")
        return

    # Check for git mode (from config or --in-place flag)
    use_in_place = in_place or config.workflow.git_mode

    # Determine output path
    if output is not None:
        # Explicit output path provided
        pass
    elif use_in_place:
        # Git mode: overwrite the original file
        output = prompt
        click.echo("(git mode: overwriting original file)")
    else:
        # Normal mode: create a new variation
        try:
            ft = FTClient(
                executable=config.ft.executable,
                timeout=config.ft.timeout,
            )
            ft_config = ft.config()

            # Get the primary prior type
            prior_type = ft_config.prior_types[0] if ft_config.prior_types else "prompt.txt"
            output_str = ft.propose(str(prompt), prior_type)
            output = Path(output_str)
        except FTError as e:
            # Fallback to simple naming
            click.echo(f"Warning: ft tool error ({e}), using fallback naming", err=True)
            stem = prompt.stem.split(".")[0]
            output = prompt.parent / f"{stem}a.prompt.txt"

        # Ensure we don't overwrite existing files (only in normal mode)
        while output.exists():
            name = output.name
            if "a" <= name[3:4] <= "z":
                # Increment variation letter
                letter = chr(ord(name[3]) + 1)
                output = output.parent / (name[:3] + letter + name[4:])
            else:
                output = output.parent / (name[:3] + "a" + name[3:])

    save_improved_prompt(improved, output)
    click.echo(f"\nSaved improved prompt to: {output}")
