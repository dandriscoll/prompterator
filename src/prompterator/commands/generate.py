"""Generate command - run a prompt through the Author LLM and capture output."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.generator import generate_from_prompt
from prompterator.core.run import create_run_dir
from prompterator.runners.llm import LLMClient, LLMError


def _resolve_content_files(config, base_dir: Path, cli_content: Path | None) -> list[Path]:
    """Resolve content file paths from CLI flag or config.

    For generate we need paths (for naming), not just text.
    """
    if cli_content is not None:
        return [cli_content]

    raw = config.directories.content
    if raw is None:
        return []

    if isinstance(raw, str):
        raw = [raw]

    paths = []
    for entry in raw:
        p = Path(entry)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            paths.append(p)
        else:
            click.echo(f"Warning: content file not found: {p}", err=True)
    return paths


@click.command("generate")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
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
    help="Base name for output files",
)
@click.option(
    "--count",
    "-n", "-s",
    type=int,
    default=1,
    help="Number of generations per content file (default: 1)",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="LLM call timeout in seconds (default: 300)",
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
    help="Show the prompt that would be sent without calling the LLM",
)
def generate_cmd(
    prompt: Path | None,
    system: str | None,
    output: Path | None,
    count: int,
    timeout: int,
    content: Path | None,
    dry_run: bool,
) -> None:
    """Run a prompt through the Author LLM and capture the output.

    PROMPT is the path to the prompt file to run. If omitted, uses the
    prompt configured in prompterator.yaml (directories.prompt).

    Content files can be specified via --content/-c or in the config under
    directories.content (single path or list). When multiple content files
    are configured, each one is run through the prompt separately.

    If the prompt contains {{INPUT}}, content is substituted in.
    Otherwise, the prompt becomes the system message and content becomes
    the user message.
    """
    config = load_config()
    base_dir = get_config_base_dir()

    if prompt is None:
        from prompterator.commands.resolve import resolve_prompt
        prompt = resolve_prompt(config, base_dir)
        if prompt is None:
            click.echo(
                "No prompt specified and none configured in prompterator.yaml.\n"
                "Either pass a prompt file or set directories.prompt in config.",
                err=True,
            )
            raise SystemExit(1)

    # Resolve content files
    content_files = _resolve_content_files(config, base_dir, content)

    # If no content files, run once with no content
    if not content_files:
        content_files = [None]

    n_content = len([f for f in content_files if f is not None])
    n_generations = count * len(content_files)
    click.echo(f"Prompt: {prompt.name}", err=True)
    if n_content:
        click.echo(f"Content files: {n_content}", err=True)
    click.echo(f"Generations: {n_generations} ({count} x {len(content_files)} content{'s' if len(content_files) != 1 else ''})", err=True)
    click.echo(f"LLM calls: {n_generations}", err=True)
    click.echo(err=True)

    if dry_run:
        prompt_text = prompt.read_text()
        for cf in content_files:
            content_text = cf.read_text() if cf else None
            if cf:
                click.echo(f"--- Content: {cf.name} ---")
            click.echo("--- Prompt (would be sent to Author LLM) ---")
            if system:
                click.echo(f"[system] {system}")
                click.echo()
            if content_text and "{{INPUT}}" in prompt_text:
                click.echo(prompt_text.replace("{{INPUT}}", content_text))
            elif content_text:
                click.echo(f"[system] {prompt_text}")
                click.echo()
                click.echo(content_text)
            else:
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

    # Determine output settings
    if output is not None:
        out_suffix = output.suffix or ".txt"
        results_dir = output.parent
    else:
        out_suffix = ".txt"
        results_dir = config.get_dir("results", base_dir)

    run_dir = create_run_dir(results_dir)

    for cf in content_files:
        content_text = cf.read_text() if cf else None

        # Output stem: -o flag > content file basename > prompt basename
        if output is not None:
            out_stem = output.stem
        elif cf is not None:
            out_stem = cf.stem.split(".")[0] + ".output"
        else:
            out_stem = prompt.stem.split(".")[0] + ".output"

        if cf and len(content_files) > 1:
            click.echo(f"Content: {cf.name}", err=True)

        for i in range(count):
            if count == 1:
                out_path = run_dir / f"{out_stem}{out_suffix}"
            else:
                out_path = run_dir / f"{out_stem}.{i + 1:03d}{out_suffix}"

            if count > 1:
                click.echo(f"  Generation {i + 1}/{count}", err=True)

            try:
                result = generate_from_prompt(
                    prompt, llm, system=system, content=content_text,
                    timeout=timeout,
                )
            except LLMError as e:
                click.echo(f"LLM error: {e}", err=True)
                raise SystemExit(1)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result + "\n")
            click.echo(f"Saved: {out_path}", err=True)
