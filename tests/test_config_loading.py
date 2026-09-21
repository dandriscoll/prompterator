"""Tests for how configuration and environment are loaded."""

import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from prompterator.cli import main
from prompterator.config.loader import load_config
from prompterator.config.schema import StackConfig


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def _run_check_in(directory: Path) -> None:
    prompt = directory / "p.prompt.md"
    prompt.write_text("a: 1\n")
    result = CliRunner().invoke(main, ["check", str(prompt)])
    assert result.exit_code == 0, result.output


def test_dotenv_does_not_replace_caller_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PROMPTERATOR_TEST_SETTING=from-file\n")
    monkeypatch.setenv("PROMPTERATOR_TEST_SETTING", "from-caller")

    _run_check_in(tmp_path)

    assert os.environ["PROMPTERATOR_TEST_SETTING"] == "from-caller"


def test_dotenv_fills_unset_variables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PROMPTERATOR_TEST_SETTING=from-file\n")
    monkeypatch.delenv("PROMPTERATOR_TEST_SETTING", raising=False)

    _run_check_in(tmp_path)

    assert os.environ["PROMPTERATOR_TEST_SETTING"] == "from-file"


# ---------------------------------------------------------------------------
# stacks.*.endpoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint", [
    None,
    "https://api.example.com/v1",
    "https://resource.openai.azure.com/",
    "http://localhost:8004/v1",
    "http://127.0.0.1:8000",
    "http://[::1]:8000/v1",
])
def test_endpoint_accepted(endpoint):
    assert StackConfig(endpoint=endpoint).endpoint == endpoint


def test_empty_endpoint_means_unset():
    assert StackConfig(endpoint="").endpoint is None


@pytest.mark.parametrize("endpoint", [
    "http://api.example.com/v1",
    "http://10.0.0.5:8000/v1",
    "ftp://api.example.com/",
    "file:///tmp/x",
    "api.example.com/v1",
    "https://",
])
def test_endpoint_rejected(endpoint):
    with pytest.raises(ValidationError, match="endpoint"):
        StackConfig(endpoint=endpoint)


def test_endpoint_rejected_when_loading_config_file(tmp_path):
    config_path = tmp_path / "prompterator.yaml"
    config_path.write_text(yaml.safe_dump({
        "stacks": {"default": {"runner": "openai", "endpoint": "http://api.example.com/v1"}},
    }))
    with pytest.raises(ValidationError, match="endpoint"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# Executables named in the config
# ---------------------------------------------------------------------------

def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _write_config(config_dir: Path, data: dict) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "prompterator.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


@pytest.mark.parametrize("key,build", [
    ("critic.script", lambda exe: {"critic": {"mode": "script", "script": str(exe)}}),
    ("naming.executable", lambda exe: {"naming": {"executable": str(exe)}}),
    ("stacks.default.runner", lambda exe: {"stacks": {"default": {"runner": str(exe)}}}),
])
def test_executable_outside_config_directory_warns(tmp_path, capsys, key, build):
    exe = _executable(tmp_path / "elsewhere" / "tool")
    config_path = _write_config(tmp_path / "project", build(exe))

    load_config(config_path)

    err = capsys.readouterr().err
    assert key in err
    assert str(exe) in err


@pytest.mark.parametrize("key,build", [
    ("critic.script", lambda exe: {"critic": {"mode": "script", "script": str(exe)}}),
    ("naming.executable", lambda exe: {"naming": {"executable": str(exe)}}),
    ("stacks.default.runner", lambda exe: {"stacks": {"default": {"runner": str(exe)}}}),
])
def test_executable_inside_config_directory_is_quiet(tmp_path, capsys, key, build):
    exe = _executable(tmp_path / "project" / "tools" / "tool")
    config_path = _write_config(tmp_path / "project", build(exe))

    load_config(config_path)

    assert capsys.readouterr().err == ""


def test_builtin_runners_and_defaults_are_quiet(tmp_path, capsys):
    config_path = _write_config(tmp_path / "project", {
        "stacks": {
            "default": {"runner": "anthropic"},
            "o": {"runner": "openai"},
            "a": {"runner": "azure-openai"},
        },
    })

    load_config(config_path)

    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# init endpoint prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("runner", ["openai", "anthropic", "azure-openai"])
def test_init_endpoint_prompt_asks_again_for_unaccepted_url(runner):
    from prompterator.commands.init import _prompt_for_endpoint

    with CliRunner().isolation(input="http://api.example.com/v1\nhttps://api.example.com/v1\n") as streams:
        endpoint, _model, _version = _prompt_for_endpoint(runner)

    assert endpoint == "https://api.example.com/v1"
    assert b"https://" in streams[0].getvalue()
