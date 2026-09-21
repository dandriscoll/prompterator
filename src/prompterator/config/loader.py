"""Configuration loading utilities."""

import shutil
from pathlib import Path

import click
import yaml

from prompterator.config.schema import BUILTIN_RUNNERS, Config

CONFIG_FILENAME = "prompterator.yaml"


def find_config_file(start_dir: Path | None = None) -> Path | None:
    """Find prompterator.yaml by searching up from start_dir."""
    if start_dir is None:
        start_dir = Path.cwd()

    current = start_dir.resolve()

    while True:
        config_path = current / CONFIG_FILENAME
        if config_path.exists():
            return config_path

        parent = current.parent
        if parent == current:  # Reached filesystem root
            return None
        current = parent


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Explicit path to config file, or None to search.

    Returns:
        Config object (defaults if no config file found).
    """
    if config_path is None:
        config_path = find_config_file()

    if config_path is None or not config_path.exists():
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    config = Config.model_validate(data)
    _warn_on_external_executables(config, config_path)
    return config


def _configured_executables(config: Config) -> list[tuple[str, str]]:
    """List (config key, value) for every program the config asks us to run."""
    items: list[tuple[str, str]] = []
    if config.critic.mode == "script" and config.critic.script:
        items.append(("critic.script", config.critic.script))
    if config.naming.executable != "naming":
        items.append(("naming.executable", config.naming.executable))
    for name, stack in config.stacks.items():
        if stack.runner not in BUILTIN_RUNNERS:
            items.append((f"stacks.{name}.runner", stack.runner))
    return items


def _resolve_executable(value: str) -> Path | None:
    """Resolve a configured program the way the runners do (path, then PATH)."""
    path = Path(value)
    if path.exists():
        return path.resolve()
    found = shutil.which(value)
    if found:
        return Path(found).resolve()
    return None


def _warn_on_external_executables(config: Config, config_path: Path) -> None:
    """Warn when the config runs a program from outside its own directory.

    prompterator.yaml can name programs to run (critic script, naming tool,
    custom LLM runners). They run with the user's permissions, so a program
    living outside the project is called out on stderr.
    """
    config_dir = config_path.resolve().parent
    for key, value in _configured_executables(config):
        resolved = _resolve_executable(value)
        if resolved is None or resolved.is_relative_to(config_dir):
            continue
        click.echo(
            f"Warning: {key} in {config_path} runs {resolved}, which is outside "
            f"{config_dir}. Make sure you trust this program.",
            err=True,
        )


def save_config(config: Config, path: Path) -> None:
    """Save configuration to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)


def get_config_base_dir(config_path: Path | None = None) -> Path:
    """Get the base directory for resolving relative paths.

    Returns the directory containing the config file, or cwd if no config.
    """
    if config_path is None:
        config_path = find_config_file()

    if config_path is not None:
        return config_path.parent

    return Path.cwd()
