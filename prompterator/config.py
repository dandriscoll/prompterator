from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ENV_TEMPLATE = """# Prompterator configuration
#
# LLM endpoints
EDITOR_LLM_ENDPOINT=
OPERATOR_LLM_ENDPOINT=

# Optional API keys
EDITOR_LLM_API_KEY=
OPERATOR_LLM_API_KEY=

# File mode: auto, git, plain
FILE_MODE=auto
# Suffix to append when writing revised files in plain mode
OUTPUT_SUFFIX=.revised

# Module that provides example parsing/evaluation
EXAMPLES_MODULE=prompt_examples

# Request timeout in seconds
REQUEST_TIMEOUT=30
"""


@dataclass(frozen=True)
class Config:
    editor_endpoint: str
    operator_endpoint: str
    editor_api_key: Optional[str]
    operator_api_key: Optional[str]
    file_mode: str
    output_suffix: str
    examples_module: Optional[str]
    request_timeout: int


class ConfigError(ValueError):
    pass


def init_env(path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ENV_TEMPLATE, encoding="utf-8")


def load_config(path: Path) -> Config:
    if not path.exists():
        raise ConfigError(f"Missing .env file at {path}. Run `prompterator init`.")
    env = load_env(path)
    editor_endpoint = _require(env, "EDITOR_LLM_ENDPOINT")
    operator_endpoint = _require(env, "OPERATOR_LLM_ENDPOINT")
    file_mode = env.get("FILE_MODE", "auto").strip() or "auto"
    output_suffix = env.get("OUTPUT_SUFFIX", ".revised").strip() or ".revised"
    examples_module = env.get("EXAMPLES_MODULE", "prompt_examples").strip() or None
    request_timeout = _parse_int(env.get("REQUEST_TIMEOUT"), default=30)

    return Config(
        editor_endpoint=editor_endpoint,
        operator_endpoint=operator_endpoint,
        editor_api_key=_blank_to_none(env.get("EDITOR_LLM_API_KEY")),
        operator_api_key=_blank_to_none(env.get("OPERATOR_LLM_API_KEY")),
        file_mode=file_mode,
        output_suffix=output_suffix,
        examples_module=examples_module,
        request_timeout=request_timeout,
    )


def load_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value.strip())
        value = _strip_quotes(value)
        values[key] = value
    return values


def _strip_inline_comment(value: str) -> str:
    if not value:
        return value
    if value[0] in {"'", '"'}:
        return value
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _require(env: Dict[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ConfigError(f"Missing required config: {key}")
    return value


def _parse_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid integer for REQUEST_TIMEOUT: {raw}") from exc
