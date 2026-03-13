"""Generate command - run a prompt through the Author LLM and capture output."""

from pathlib import Path

import click

from prompterator.config.loader import load_config
from prompterator.core.generator import generate_from_prompt
from prompterator.core.run import create_run_dir
from prompterator.runners.llm import LLMClient, LLMError


@click.command("generate")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--system",
    type=str,
    default=None,
    help="System prompt to use (default: none)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write output to file instead of stdout",
)
@click.option(
    "--count",
    "-n",
    type=int,
    default=1,
    help="Number of generations to produce (default: 1)",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="LLM call timeout in seconds (default: 300)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the prompt that would be sent without calling the LLM",
)
def generate_cmd(
    prompt: Path,
    system: str | None,
    output: Path | None,
    count: int,
    timeout: int,
    dry_run: bool,
) -> None:
    """Run a prompt through the Author LLM and capture the output.

    PROMPT is the path to the prompt file to run.

    With --output/-o, writes each generation to a numbered file
    (e.g. output.001.txt, output.002.txt). Without it, prints to stdout.
    """
    config = load_config()

    if dry_run:
        prompt_text = prompt.read_text()
        click.echo("--- Prompt (would be sent to Author LLM) ---")
        if system:
            click.echo(f"[system] {system}")
            click.echo()
        click.echo(prompt_text)
        click.echo("--- End ---")
        return

    # Initialize Author LLM client
    try:
        llm = LLMClient(**config.resolve_role("author"))
    except LLMError as e:
        click.echo(f"Author LLM error: {e}", err=True)
        raise SystemExit(1)

    from prompterator.runners.llm import debug_context
    debug_context("generate")

    for i in range(count):
        if count > 1:
            click.echo(f"--- Generation {i + 1}/{count} ---", err=True)

        try:
            result = generate_from_prompt(
                prompt, llm, system=system, timeout=timeout,
            )
        except LLMError as e:
            click.echo(f"LLM error: {e}", err=True)
            raise SystemExit(1)

        if output is not None:
            # Create a run directory for output isolation
            if i == 0:
                run_dir = create_run_dir(output.parent)

            if count == 1:
                out_path = run_dir / output.name
            else:
                stem = output.stem
                suffix = output.suffix or ".txt"
                out_path = run_dir / f"{stem}.{i + 1:03d}{suffix}"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result + "\n")
            click.echo(f"Saved: {out_path}", err=True)
        else:
            click.echo(result)
            if count > 1 and i < count - 1:
                click.echo()
