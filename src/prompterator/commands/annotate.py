"""Annotate command - create .mb feedback files without writing boilerplate."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.run import create_run_dir
from prompterator.core.annotator import (
    build_editor_template,
    build_mb_content,
    derive_mb_path,
    parse_editor_result,
)


def _read_stdin_observations() -> list[str]:
    """Read observations from stdin, one per line."""
    observations = []
    for line in sys.stdin:
        stripped = line.strip()
        if stripped:
            observations.append(stripped)
    return observations


def _edit_interactively(template: str) -> list[str]:
    """Open $EDITOR with a template and return parsed observations."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "vi")

    with tempfile.NamedTemporaryFile(suffix=".mb.txt", mode="w", delete=False) as f:
        f.write(template)
        tmp_path = f.name

    try:
        subprocess.run([editor, tmp_path], check=True)
        with open(tmp_path) as f:
            result = f.read()
        return parse_editor_result(result)
    except subprocess.CalledProcessError:
        return []
    finally:
        os.unlink(tmp_path)


@click.command("annotate")
@click.argument("source", type=str)
@click.option(
    "--prior",
    "priors",
    type=str,
    multiple=True,
    required=True,
    help="Prior file (prompt, input data, etc.). Repeatable.",
)
@click.option(
    "-m",
    "messages",
    type=str,
    multiple=True,
    help="Feedback observation. Repeatable for multiple observations.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output .mb file path (default: derived from source name)",
)
@click.option(
    "--append",
    "-a",
    is_flag=True,
    help="Append to existing .mb file instead of overwriting",
)
@click.option(
    "--edit",
    "-e",
    is_flag=True,
    help="Open $EDITOR to write observations interactively",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the .mb content without writing to disk",
)
def annotate_cmd(
    source: str,
    priors: tuple[str, ...],
    messages: tuple[str, ...],
    output: Path | None,
    append: bool,
    edit: bool,
    dry_run: bool,
) -> None:
    """Create a .mb feedback file from a source and observations.

    SOURCE is the path to the output being reviewed (text file, image, etc.).

    \b
    Three ways to provide observations:
      -m "text"     Inline (repeatable for multiple observations)
      --edit / -e   Open $EDITOR to write observations interactively
      stdin         Pipe observations, one per line

    \b
    Examples:
      prompterator annotate outputs/001-r1.out.md \\
        --prior improve-todo.prompt.md --prior 001.todoosy.md \\
        -m "opens with conversational preamble" \\
        -m "replaced checkboxes with priority sections"

      prompterator annotate outputs/002-r1.out.md \\
        --prior improve-todo.prompt.md --prior 002.todoosy.md \\
        -e

      echo "chatbot sign-off at the end" | prompterator annotate outputs/003-r1.out.md \\
        --prior improve-todo.prompt.md
    """
    config = load_config()
    base_dir = get_config_base_dir()
    prior_list = list(priors)

    # Collect observations from all input methods
    observations: list[str] = list(messages)

    if edit:
        template = build_editor_template(source, prior_list)
        editor_obs = _edit_interactively(template)
        observations.extend(editor_obs)
    elif not messages and not sys.stdin.isatty():
        # Read from stdin if no -m flags and stdin is piped
        observations.extend(_read_stdin_observations())

    if not observations:
        click.echo("No observations provided. Use -m, --edit, or pipe to stdin.", err=True)
        raise SystemExit(1)

    # Build markback content
    content = build_mb_content(source, prior_list, observations)

    if dry_run:
        click.echo(content, nl=False)
        return

    # Determine output path
    if output is None:
        feedback_dir = config.get_dir("feedback", base_dir)
        run_dir = create_run_dir(feedback_dir)
        output = derive_mb_path(source, run_dir)

    # Write or append
    output.parent.mkdir(parents=True, exist_ok=True)

    if append and output.exists():
        existing = output.read_text()
        # Ensure separator between existing content and new blocks
        if existing and not existing.endswith("\n"):
            existing += "\n"
        content = existing + "\n---\n\n" + content

    output.write_text(content)

    click.echo(f"{'Appended to' if append else 'Saved'}: {output}", err=True)
    click.echo(f"Observations: {len(observations)}", err=True)
