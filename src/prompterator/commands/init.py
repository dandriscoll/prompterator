"""Init command - create prompterator.yaml config."""

from pathlib import Path

import click

from prompterator.config.loader import CONFIG_FILENAME, save_config
from prompterator.config.schema import Config


@click.command("init")
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing config file",
)
def init_cmd(force: bool) -> None:
    """Create a new prompterator.yaml configuration file."""
    config_path = Path.cwd() / CONFIG_FILENAME

    if config_path.exists() and not force:
        click.echo(f"Config file already exists: {config_path}")
        click.echo("Use --force to overwrite.")
        raise SystemExit(1)

    config = Config()
    save_config(config, config_path)

    click.echo(f"Created {CONFIG_FILENAME}")
    click.echo()
    click.echo("Default directories:")
    click.echo(f"  prompts:  {config.directories.prompts}")
    click.echo(f"  feedback: {config.directories.feedback}")
    click.echo(f"  issues:   {config.directories.issues}")
    click.echo(f"  evals:    {config.directories.evals}")
    click.echo(f"  results:  {config.directories.results}")
