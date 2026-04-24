"""Check command - validate that a prompt file parses as YAML.

Useful for prompts authored as YAML (plain or `---`-delimited frontmatter
style), where a malformed edit from the editor LLM would otherwise only
surface at the next author run. Plain-text prompts are out of scope — this
command is for YAML-shaped prompts by convention.
"""

from pathlib import Path

import click
import yaml

from prompterator.commands.resolve import resolve_prompt
from prompterator.config.loader import get_config_base_dir, load_config


def _extract_yaml_body(text: str) -> tuple[str, bool]:
    """Return (yaml_body, was_frontmatter).

    If the file is wrapped in `---…---` (frontmatter style — what the
    content files and lora.prompt.md use), strip the wrappers so the
    middle parses as a single YAML document. Otherwise return as-is.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return text, False

    lines = stripped.splitlines()
    if lines[0].strip() != "---":
        return text, False

    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        # opening --- without a closing one: yaml.safe_load can handle this
        return "\n".join(lines[1:]), True

    return "\n".join(lines[1:end]), True


@click.command("check")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
def check_cmd(prompt: Path | None) -> None:
    """Validate that a prompt file parses as YAML.

    PROMPT is the path to the prompt file (optional — resolved from config
    if omitted). Handles both plain YAML and `---`-delimited frontmatter.
    Exits 0 on success, 1 on any parse failure.
    """
    config = load_config()
    base_dir = get_config_base_dir()

    if prompt is None:
        prompt = resolve_prompt(config, base_dir)
        if prompt is None:
            click.echo(
                "Error: no prompt specified and none configured under "
                "directories.prompt.",
                err=True,
            )
            raise SystemExit(1)

    text = prompt.read_text()
    body, was_frontmatter = _extract_yaml_body(text)

    try:
        parsed = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        click.echo(f"{prompt}: invalid YAML", err=True)
        click.echo(str(exc), err=True)
        raise SystemExit(1)

    if parsed is None:
        click.echo(f"{prompt}: parsed empty (no YAML content)", err=True)
        raise SystemExit(1)

    if not isinstance(parsed, dict):
        click.echo(
            f"{prompt}: parsed as {type(parsed).__name__}, expected a mapping",
            err=True,
        )
        raise SystemExit(1)

    style = "frontmatter (--- delimited)" if was_frontmatter else "plain YAML"
    click.echo(f"{prompt}: OK ({style})")
    click.echo(f"  keys: {', '.join(sorted(parsed.keys()))}")
