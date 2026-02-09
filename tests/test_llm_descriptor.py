"""Tests for the LLM runner Boutiques descriptor contract."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BIN_DIR = Path(__file__).parent.parent / "bin"
DESCRIPTOR_FILE = BIN_DIR / "llm-runner.boutiques.json"
RUNNERS = ["llm-anthropic", "llm-openai", "llm-azure-openai"]


class TestDescriptorFile:
    """Tests for the standalone descriptor file."""

    def test_descriptor_is_valid_json(self):
        text = DESCRIPTOR_FILE.read_text()
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_descriptor_has_required_keys(self):
        data = json.loads(DESCRIPTOR_FILE.read_text())
        assert "name" in data
        assert "inputs" in data
        assert isinstance(data["inputs"], list)

    def test_descriptor_has_expected_inputs(self):
        data = json.loads(DESCRIPTOR_FILE.read_text())
        input_ids = {inp["id"] for inp in data["inputs"]}
        expected = {"system", "temperature", "max_tokens", "model", "endpoint", "api_version", "descriptor"}
        assert expected == input_ids


class TestRunnerDescriptorFlag:
    """Tests that each built-in runner outputs the descriptor with --descriptor."""

    @pytest.mark.parametrize("runner", RUNNERS)
    def test_runner_descriptor_matches_standalone(self, runner):
        runner_path = BIN_DIR / runner
        result = subprocess.run(
            [sys.executable, str(runner_path), "--descriptor"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        runner_output = json.loads(result.stdout)
        standalone = json.loads(DESCRIPTOR_FILE.read_text())
        assert runner_output == standalone

    @pytest.mark.parametrize("runner", RUNNERS)
    def test_runner_descriptor_is_valid_json(self, runner):
        runner_path = BIN_DIR / runner
        result = subprocess.run(
            [sys.executable, str(runner_path), "--descriptor"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["name"] == "llm-runner"


class TestLLMClientDescriptor:
    """Tests for LLMClient.descriptor() method."""

    def test_descriptor_returns_parsed_json(self):
        from prompterator.runners.llm import LLMClient

        client = LLMClient(runner="anthropic")
        desc = client.descriptor()
        assert isinstance(desc, dict)
        assert desc["name"] == "llm-runner"
        assert "inputs" in desc

    def test_descriptor_input_ids(self):
        from prompterator.runners.llm import LLMClient

        client = LLMClient(runner="openai")
        desc = client.descriptor()
        input_ids = {inp["id"] for inp in desc["inputs"]}
        assert "descriptor" in input_ids
        assert "system" in input_ids
