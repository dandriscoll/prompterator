"""Init command - create prompterator.yaml config."""

from pathlib import Path

import click

from prompterator.config.loader import CONFIG_FILENAME, save_config
from prompterator.config.schema import AuthorConfig, Config, CriticConfig, EditorConfig


RUNNER_CHOICES = ["azure-openai", "anthropic", "openai", "custom"]


def _prompt_for_runner() -> str:
    """Prompt user to select an LLM runner."""
    click.echo("Which LLM runner will you use?")
    click.echo()
    click.echo("  1) azure-openai - Azure OpenAI API (requires AZURE_OPENAI_API_KEY)")
    click.echo("  2) anthropic    - Claude API (requires ANTHROPIC_API_KEY)")
    click.echo("  3) openai       - OpenAI API (requires OPENAI_API_KEY)")
    click.echo("  4) custom       - Custom runner script")
    click.echo()

    choice = click.prompt(
        "Select runner",
        type=click.Choice(["1", "2", "3", "4"]),
        default="1",
    )

    runner_map = {"1": "azure-openai", "2": "anthropic", "3": "openai", "4": "custom"}
    runner = runner_map[choice]

    if runner == "custom":
        runner = click.prompt("Path to custom runner script")

    return runner


def _get_default_model(runner: str) -> str:
    """Get default model for a runner."""
    defaults = {
        "azure-openai": "gpt-4o",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
    }
    return defaults.get(runner, "")


def _prompt_for_endpoint(runner: str) -> str | None:
    """Prompt for endpoint configuration."""
    if runner == "azure-openai":
        click.echo()
        click.echo("Azure OpenAI requires an endpoint URL.")
        click.echo("Leave blank to use AZURE_OPENAI_ENDPOINT environment variable.")
        endpoint = click.prompt("Endpoint URL", default="", show_default=False)
        return endpoint if endpoint else None
    elif runner in ("openai", "anthropic"):
        click.echo()
        click.echo(f"Custom endpoint (leave blank for default {runner.title()} API):")
        endpoint = click.prompt("Endpoint URL", default="", show_default=False)
        return endpoint if endpoint else None
    return None


def _prompt_for_model(runner: str) -> str | None:
    """Prompt for model configuration."""
    if runner in ("azure-openai", "openai", "anthropic"):
        default = _get_default_model(runner)
        click.echo()
        model = click.prompt("Model/deployment name", default=default)
        return model
    return None


def _prompt_for_api_version(runner: str) -> str | None:
    """Prompt for API version (Azure only)."""
    if runner == "azure-openai":
        click.echo()
        api_version = click.prompt("API version", default="2024-02-01")
        return api_version
    return None


def _create_role_configs(
    runner: str,
    model: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
) -> tuple[AuthorConfig, EditorConfig, CriticConfig]:
    """Create LLM configs for all roles based on runner choice."""
    common = {
        "runner": runner,
        "max_tokens": 4096,
    }
    if model:
        common["model"] = model
    if endpoint:
        common["endpoint"] = endpoint
    if api_version:
        common["api_version"] = api_version

    return (
        AuthorConfig(**common, temperature=0.7),
        EditorConfig(**common, temperature=0.7),
        CriticConfig(**common, temperature=0.3),
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
    endpoint = _prompt_for_endpoint(runner)
    model = _prompt_for_model(runner)
    api_version = _prompt_for_api_version(runner)
    author_config, editor_config, critic_config = _create_role_configs(
        runner, model=model, endpoint=endpoint, api_version=api_version
    )

    config = Config(author=author_config, editor=editor_config, critic=critic_config)
    save_config(config, config_path)

    click.echo()
    click.echo(f"Created {CONFIG_FILENAME}")
    click.echo()
    click.echo("LLM Configuration:")
    click.echo(f"  Runner:   {config.author.runner}")
    if config.author.model:
        click.echo(f"  Model:    {config.author.model}")
    if config.author.endpoint:
        click.echo(f"  Endpoint: {config.author.endpoint}")
    if config.author.api_version:
        click.echo(f"  API Ver:  {config.author.api_version}")
    click.echo()
    click.echo("LLM Roles:")
    click.echo(f"  Author: temp={config.author.temperature}")
    click.echo(f"  Editor: temp={config.editor.temperature}")
    click.echo(f"  Critic: temp={config.critic.temperature}")
    click.echo()
    click.echo("Default directories:")
    click.echo(f"  prompts:  {config.directories.prompts}")
    click.echo(f"  feedback: {config.directories.feedback}")
    click.echo(f"  issues:   {config.directories.issues}")
    click.echo(f"  evals:    {config.directories.evals}")
    click.echo(f"  results:  {config.directories.results}")
