"""Configuration loading utilities."""

from pathlib import Path

import yaml

from prompterator.config.schema import Config

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

    return Config.model_validate(data)


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
