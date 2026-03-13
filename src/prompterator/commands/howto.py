"""Howto command - print the HOWTO guide for setting up prompterator."""

from importlib import resources

import click


@click.command("howto")
def howto_cmd() -> None:
    """Print the HOWTO setup guide (designed to be piped to an LLM agent)."""
    howto_path = resources.files("prompterator").joinpath("HOWTO.md")
    click.echo(howto_path.read_text(encoding="utf-8"))
