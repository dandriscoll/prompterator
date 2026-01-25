"""Init command - create prompterator.yaml config."""

from pathlib import Path

import click

from prompterator.config.loader import CONFIG_FILENAME, save_config
from prompterator.config.schema import AuthorConfig, Config, CriticConfig, EditorConfig


RUNNER_CHOICES = ["anthropic", "openai", "custom"]


def _prompt_for_runner() -> str:
    """Prompt user to select an LLM runner."""
    click.echo("Which LLM runner will you use?")
    click.echo()
    click.echo("  1) anthropic - Claude API (requires ANTHROPIC_API_KEY)")
    click.echo("  2) openai    - OpenAI API (requires OPENAI_API_KEY)")
    click.echo("  3) custom    - Custom runner script")
    click.echo()

    choice = click.prompt(
        "Select runner",
        type=click.Choice(["1", "2", "3"]),
        default="1",
    )

    runner_map = {"1": "anthropic", "2": "openai", "3": "custom"}
    runner = runner_map[choice]

    if runner == "custom":
        runner = click.prompt("Path to custom runner script")

    return runner


def _create_role_configs(
    runner: str,
) -> tuple[AuthorConfig, EditorConfig, CriticConfig]:
    """Create LLM configs for all roles based on runner choice."""
    return (
        AuthorConfig(runner=runner, temperature=0.7, max_tokens=4096),
        EditorConfig(runner=runner, temperature=0.7, max_tokens=4096),
        CriticConfig(runner=runner, temperature=0.3, max_tokens=4096),
    )


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

    runner = _prompt_for_runner()
    author_config, editor_config, critic_config = _create_role_configs(runner)

    config = Config(author=author_config, editor=editor_config, critic=critic_config)
    save_config(config, config_path)

    click.echo()
    click.echo(f"Created {CONFIG_FILENAME}")
    click.echo()
    click.echo("LLM Roles:")
    click.echo(f"  Author: {config.author.runner} (temp={config.author.temperature})")
    click.echo(f"  Editor: {config.editor.runner} (temp={config.editor.temperature})")
    click.echo(f"  Critic: {config.critic.runner} (temp={config.critic.temperature})")
    click.echo()
    click.echo("Default directories:")
    click.echo(f"  prompts:  {config.directories.prompts}")
    click.echo(f"  feedback: {config.directories.feedback}")
    click.echo(f"  issues:   {config.directories.issues}")
    click.echo(f"  evals:    {config.directories.evals}")
    click.echo(f"  results:  {config.directories.results}")
