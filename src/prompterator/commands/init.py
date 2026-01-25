"""Init command - create prompterator.yaml config."""

from pathlib import Path

import click

from prompterator.config.loader import CONFIG_FILENAME, save_config
from prompterator.config.schema import Config, LLMConfig


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


def _create_llm_config(runner: str) -> LLMConfig:
    """Create LLM config based on runner choice."""
    if runner == "anthropic":
        return LLMConfig(
            runner="anthropic",
            temperature=0.7,
            max_tokens=4096,
        )
    elif runner == "openai":
        return LLMConfig(
            runner="openai",
            temperature=0.7,
            max_tokens=4096,
        )
    else:
        # Custom runner
        return LLMConfig(
            runner=runner,
            temperature=0.7,
            max_tokens=4096,
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
    llm_config = _create_llm_config(runner)

    config = Config(llm=llm_config)
    save_config(config, config_path)

    click.echo()
    click.echo(f"Created {CONFIG_FILENAME}")
    click.echo()
    click.echo("LLM configuration:")
    click.echo(f"  runner:      {config.llm.runner}")
    click.echo(f"  temperature: {config.llm.temperature}")
    click.echo(f"  max_tokens:  {config.llm.max_tokens}")
    click.echo()
    click.echo("Default directories:")
    click.echo(f"  prompts:  {config.directories.prompts}")
    click.echo(f"  feedback: {config.directories.feedback}")
    click.echo(f"  issues:   {config.directories.issues}")
    click.echo(f"  evals:    {config.directories.evals}")
    click.echo(f"  results:  {config.directories.results}")
