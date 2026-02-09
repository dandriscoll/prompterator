"""Init command - create prompterator.yaml config."""

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import click

from prompterator.config.loader import CONFIG_FILENAME, save_config
from prompterator.config.schema import (
    AuthorConfig,
    Config,
    CriticConfig,
    EditorConfig,
    StackConfig,
)


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


def _parse_azure_url(url: str) -> dict[str, str | None]:
    """Parse a full Azure OpenAI chat completions URL.

    Detects URLs like:
        https://resource.openai.azure.com/openai/deployments/gpt-5.1-chat/chat/completions?api-version=2024-02-01

    Returns a dict with keys: endpoint, model, api_version.
    If the URL doesn't match the expected pattern, returns the url as-is for endpoint.
    """
    parsed = urlparse(url)
    match = re.match(
        r"/openai/deployments/([^/]+)/chat/completions",
        parsed.path,
    )
    if not match:
        return {"endpoint": url, "model": None, "api_version": None}

    base = f"{parsed.scheme}://{parsed.netloc}/"
    model = match.group(1)
    qs = parse_qs(parsed.query)
    api_version = qs.get("api-version", [None])[0]
    return {"endpoint": base, "model": model, "api_version": api_version}


def _prompt_for_endpoint(runner: str) -> tuple[str | None, str | None, str | None]:
    """Prompt for endpoint configuration.

    Returns (endpoint, parsed_model, parsed_api_version).
    parsed_model and parsed_api_version are only set for azure-openai full URLs.
    """
    if runner == "azure-openai":
        click.echo()
        click.echo("Azure OpenAI requires an endpoint URL.")
        click.echo("You may paste a full chat completions URL or just the base endpoint.")
        click.echo("Leave blank to use AZURE_OPENAI_ENDPOINT environment variable.")
        endpoint = click.prompt("Endpoint URL", default="", show_default=False)
        if not endpoint:
            return None, None, None
        parsed = _parse_azure_url(endpoint)
        return parsed["endpoint"], parsed["model"], parsed["api_version"]
    elif runner in ("openai", "anthropic"):
        click.echo()
        click.echo(f"Custom endpoint (leave blank for default {runner.title()} API):")
        endpoint = click.prompt("Endpoint URL", default="", show_default=False)
        return (endpoint if endpoint else None), None, None
    return None, None, None


def _prompt_for_model(runner: str, default_override: str | None = None) -> str | None:
    """Prompt for model configuration."""
    if runner in ("azure-openai", "openai", "anthropic"):
        default = default_override or _get_default_model(runner)
        click.echo()
        model = click.prompt("Model/deployment name", default=default)
        return model
    return None


def _prompt_for_api_version(runner: str, default_override: str | None = None) -> str | None:
    """Prompt for API version (Azure only)."""
    if runner == "azure-openai":
        default = default_override or "2024-02-01"
        click.echo()
        api_version = click.prompt("API version", default=default)
        return api_version
    return None


def _prompt_for_stack(stack_number: int) -> tuple[str, StackConfig]:
    """Prompt user to define a single stack. Returns (name, StackConfig)."""
    click.echo()
    click.echo(f"--- Stack {stack_number} ---")
    name = click.prompt("Stack name", default=f"stack-{stack_number}")

    runner = _prompt_for_runner()
    endpoint, parsed_model, parsed_api_version = _prompt_for_endpoint(runner)
    model = _prompt_for_model(runner, default_override=parsed_model)
    api_version = _prompt_for_api_version(runner, default_override=parsed_api_version)

    stack = StackConfig(
        runner=runner,
        model=model,
        endpoint=endpoint,
        api_version=api_version,
    )
    return name, stack


def _prompt_for_stacks() -> dict[str, StackConfig]:
    """Prompt user to define one or more named stacks."""
    click.echo()
    click.echo("Define your LLM stacks (connection configurations).")
    click.echo("You can define multiple stacks and assign them to roles.")

    stacks: dict[str, StackConfig] = {}
    stack_number = 1

    while True:
        name, stack = _prompt_for_stack(stack_number)
        stacks[name] = stack
        stack_number += 1

        click.echo()
        if not click.confirm("Add another stack?", default=False):
            break

    return stacks


ROLE_DESCRIPTIONS = {
    "author": "generates new prompt drafts from priors",
    "editor": "revises prompts based on feedback and issues",
    "critic": "evaluates prompts against rubrics and assertions",
}


def _prompt_for_role_stack(role_name: str, stack_names: list[str], default: str) -> str:
    """Prompt user to pick a stack for a role."""
    desc = ROLE_DESCRIPTIONS.get(role_name, "")
    if len(stack_names) == 1:
        click.echo()
        click.echo(f"{role_name.title()} ({desc}) → using stack '{stack_names[0]}'")
        return stack_names[0]

    click.echo()
    click.echo(f"Which stack should the {role_name} use? ({desc})")
    for i, name in enumerate(stack_names, 1):
        click.echo(f"  {i}) {name}")

    choices = [str(i) for i in range(1, len(stack_names) + 1)]
    default_idx = str(stack_names.index(default) + 1) if default in stack_names else "1"
    choice = click.prompt(
        f"Select stack for {role_name}",
        type=click.Choice(choices),
        default=default_idx,
    )
    return stack_names[int(choice) - 1]


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

    stacks = _prompt_for_stacks()
    stack_names = list(stacks.keys())
    first_stack = stack_names[0]

    author_stack = _prompt_for_role_stack("author", stack_names, first_stack)
    editor_stack = _prompt_for_role_stack("editor", stack_names, first_stack)
    critic_stack = _prompt_for_role_stack("critic", stack_names, first_stack)

    author_config = AuthorConfig(stack=author_stack, temperature=0.7, max_tokens=4096)
    editor_config = EditorConfig(stack=editor_stack, temperature=0.7, max_tokens=4096)
    critic_config = CriticConfig(stack=critic_stack, temperature=0.3, max_tokens=4096)

    config = Config(
        stacks=stacks,
        author=author_config,
        editor=editor_config,
        critic=critic_config,
    )
    save_config(config, config_path)

    click.echo()
    click.echo(f"Created {CONFIG_FILENAME}")
    click.echo()
    click.echo("Stacks:")
    for name, stack in stacks.items():
        parts = [f"runner={stack.runner}"]
        if stack.model:
            parts.append(f"model={stack.model}")
        if stack.endpoint:
            parts.append(f"endpoint={stack.endpoint}")
        click.echo(f"  {name}: {', '.join(parts)}")
    click.echo()
    click.echo("LLM Roles:")
    click.echo(f"  Author: stack={config.author.stack}, temp={config.author.temperature}")
    click.echo(f"  Editor: stack={config.editor.stack}, temp={config.editor.temperature}")
    click.echo(f"  Critic: stack={config.critic.stack}, temp={config.critic.temperature}")
    click.echo()
    click.echo("Default directories:")
    click.echo(f"  prompts:  {config.directories.prompts}")
    click.echo(f"  feedback: {config.directories.feedback}")
    click.echo(f"  issues:   {config.directories.issues}")
    click.echo(f"  evals:    {config.directories.evals}")
    click.echo(f"  results:  {config.directories.results}")

    # Collect env var reminders across all stacks
    env_vars = set()
    for stack in stacks.values():
        env_var = {
            "azure-openai": "AZURE_OPENAI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(stack.runner)
        if env_var:
            env_vars.add(env_var)
        if stack.runner == "azure-openai" and not stack.endpoint:
            env_vars.add("AZURE_OPENAI_ENDPOINT")

    if env_vars:
        click.echo()
        click.echo("Remember to set the following environment variables:")
        for var in sorted(env_vars):
            click.echo(f"  {var}")
