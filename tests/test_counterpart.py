"""Tests for counterpart LLM multi-turn dialog."""

import tempfile
from pathlib import Path

from prompterator.core.eval_runner import (
    _build_conversation_prompt,
    run_all_evals,
    run_multi_turn_dialog,
)
from prompterator.models.eval import Eval, EvalFile, EvalRubric

from tests.conftest import MockLLMClient


# ---------------------------------------------------------------------------
# _build_conversation_prompt
# ---------------------------------------------------------------------------

def test_conversation_prompt_no_history():
    """With no history, returns the message unchanged."""
    result = _build_conversation_prompt("Hello", [])
    assert result == "Hello"


def test_conversation_prompt_with_history():
    """With history, includes previous responses and labels current message."""
    result = _build_conversation_prompt("New question", ["First answer", "Second answer"])
    assert "[Previous response 1]" in result
    assert "First answer" in result
    assert "[Previous response 2]" in result
    assert "Second answer" in result
    assert "[Current message to respond to]" in result
    assert "New question" in result


# ---------------------------------------------------------------------------
# run_multi_turn_dialog
# ---------------------------------------------------------------------------

def test_multi_turn_dialog_basic():
    """Basic 2-turn dialog produces correct transcript structure."""
    author = MockLLMClient(responses=[
        "Author turn 1 response",
        "Author turn 2 response",
    ])
    counterpart = MockLLMClient(responses=[
        "Counterpart turn 1 response",
    ])

    transcript = run_multi_turn_dialog(
        "You are helpful.",
        author,
        counterpart,
        "You are a user asking questions.",
        turns=2,
    )

    assert "[Author turn 1]" in transcript
    assert "Author turn 1 response" in transcript
    assert "[Counterpart turn 1]" in transcript
    assert "Counterpart turn 1 response" in transcript
    assert "[Author turn 2]" in transcript
    assert "Author turn 2 response" in transcript


def test_multi_turn_dialog_single_turn():
    """Single turn produces only author output, no counterpart."""
    author = MockLLMClient(responses=["Solo response"])
    counterpart = MockLLMClient(responses=[])

    transcript = run_multi_turn_dialog(
        "You are helpful.",
        author,
        counterpart,
        "You are a user.",
        turns=1,
    )

    assert "[Author turn 1]" in transcript
    assert "Solo response" in transcript
    assert "Counterpart" not in transcript


def test_multi_turn_dialog_with_content():
    """Content is used as the initial input to the author."""
    author = MockLLMClient(responses=["Response about topic"])
    counterpart = MockLLMClient(responses=[])

    transcript = run_multi_turn_dialog(
        "You are helpful.",
        author,
        counterpart,
        "Directions for counterpart.",
        content_text="Tell me about Python",
        turns=1,
    )

    # Author should have received the content
    assert len(author.calls) == 1
    assert "Tell me about Python" in author.calls[0]["prompt"]


def test_multi_turn_dialog_three_turns():
    """Three turns: author-counterpart-author-counterpart-author."""
    author = MockLLMClient(responses=["A1", "A2", "A3"])
    counterpart = MockLLMClient(responses=["C1", "C2"])

    transcript = run_multi_turn_dialog(
        "System prompt",
        author,
        counterpart,
        "Counterpart directions",
        turns=3,
    )

    assert "[Author turn 1]" in transcript
    assert "[Counterpart turn 1]" in transcript
    assert "[Author turn 2]" in transcript
    assert "[Counterpart turn 2]" in transcript
    assert "[Author turn 3]" in transcript
    # No counterpart turn 3 (counterpart gets turns-1)
    assert "Counterpart turn 3" not in transcript


def test_multi_turn_dialog_author_calls_count():
    """Author gets called once per turn."""
    author = MockLLMClient(responses=["A1", "A2", "A3"])
    counterpart = MockLLMClient(responses=["C1", "C2"])

    run_multi_turn_dialog(
        "System prompt",
        author,
        counterpart,
        "Directions",
        turns=3,
    )

    assert len(author.calls) == 3
    assert len(counterpart.calls) == 2


def test_multi_turn_dialog_counterpart_receives_author_response():
    """Counterpart receives the author's response as input."""
    author = MockLLMClient(responses=["Author says hello", "Author says bye"])
    counterpart = MockLLMClient(responses=["User reply"])

    run_multi_turn_dialog(
        "System prompt",
        author,
        counterpart,
        "Be a user",
        turns=2,
    )

    # Counterpart should receive author's first response
    assert "Author says hello" in counterpart.calls[0]["prompt"]
    # Counterpart should have directions as system prompt
    assert counterpart.calls[0]["system"] == "Be a user"


# ---------------------------------------------------------------------------
# run_all_evals with counterpart
# ---------------------------------------------------------------------------

def _make_eval_file(n_evals=1):
    return EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[
            Eval(
                id=f"eval-{i:02d}",
                type="rubric",
                rubric=EvalRubric(criteria=[f"Check criterion {i}"]),
            )
            for i in range(1, n_evals + 1)
        ],
    )


def test_run_all_evals_with_counterpart():
    """Multi-turn dialog is used when counterpart is provided."""
    eval_file = _make_eval_file(1)

    # Author: 2 turns, counterpart: 1 turn, critic: 1 ensemble
    # Total LLM calls: 2 author + 1 counterpart + 1 critic = 4
    author = MockLLMClient(responses=[
        "Author turn 1",
        "Author turn 2",
    ])
    counterpart = MockLLMClient(responses=[
        "Counterpart turn 1",
    ])
    critic = MockLLMClient(responses=[
        "CRITERION 1: PASS\nREASON 1: Good dialog\nOVERALL: PASS",
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Be a helpful assistant.")
        tmp = Path(f.name)

    try:
        result_file = run_all_evals(
            eval_file, tmp, critic,
            author_llm=author,
            ensemble=1,
            counterpart_llm=counterpart,
            counterpart_directions=["You are a curious user."],
            dialog_turns=2,
        )
    finally:
        tmp.unlink()

    assert len(result_file.results) == 1
    assert result_file.results[0].passed
    # The generated output should be the dialog transcript
    assert "[Author turn 1]" in result_file.generated_output
    assert "[Counterpart turn 1]" in result_file.generated_output
    assert "[Author turn 2]" in result_file.generated_output


def test_run_all_evals_no_counterpart_unchanged():
    """Without counterpart, behavior is unchanged (single-shot author)."""
    eval_file = _make_eval_file(1)

    author = MockLLMClient(responses=["Single response"])
    critic = MockLLMClient(responses=[
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Be helpful.")
        tmp = Path(f.name)

    try:
        result_file = run_all_evals(
            eval_file, tmp, critic,
            author_llm=author,
            ensemble=1,
        )
    finally:
        tmp.unlink()

    assert result_file.generated_output == "Single response"
    assert "[Author turn" not in result_file.generated_output


def test_run_all_evals_counterpart_directions_cycle():
    """Directions cycle when fewer than content files."""
    eval_file = _make_eval_file(1)

    # 2 content files but only 1 directions → it cycles
    author = MockLLMClient(responses=[
        "A1-c1", "A1-c2",
    ])
    counterpart = MockLLMClient(responses=[
        "C1-c1", "C1-c2",
    ])
    critic = MockLLMClient(responses=[
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
        "CRITERION 1: PASS\nREASON 1: ok\nOVERALL: PASS",
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Be helpful.")
        tmp = Path(f.name)

    try:
        result_file = run_all_evals(
            eval_file, tmp, critic,
            author_llm=author,
            content_texts=["Content 1", "Content 2"],
            ensemble=1,
            counterpart_llm=counterpart,
            counterpart_directions=["Be a user."],
            dialog_turns=1,
        )
    finally:
        tmp.unlink()

    assert len(result_file.results) == 1
    # Both content files should have been processed
    assert len(author.calls) == 2


# ---------------------------------------------------------------------------
# Config: CounterpartConfig
# ---------------------------------------------------------------------------

def test_counterpart_config_defaults():
    """CounterpartConfig has sensible defaults."""
    from prompterator.config.schema import Config

    config = Config()
    assert config.counterpart.temperature == 0.7
    assert config.counterpart.stack == "default"
    assert config.counterpart.max_tokens == 4096


def test_counterpart_config_resolve_role():
    """resolve_role works for counterpart."""
    from prompterator.config.schema import Config

    config = Config()
    role_dict = config.resolve_role("counterpart")
    assert role_dict["runner"] == "anthropic"
    assert role_dict["temperature"] == 0.7


def test_counterpart_config_to_yaml_when_configured():
    """counterpart appears in YAML only when directories.counterpart is set."""
    from prompterator.config.schema import Config

    config = Config(directories={"counterpart": "scripts/user.txt"})
    d = config.to_yaml_dict()
    assert "counterpart" in d
    assert d["counterpart"]["temperature"] == 0.7
    assert d["directories"]["counterpart"] == "scripts/user.txt"


def test_counterpart_config_to_yaml_omitted_by_default():
    """counterpart is omitted from YAML when not configured."""
    from prompterator.config.schema import Config

    config = Config()
    d = config.to_yaml_dict()
    assert "counterpart" not in d
    assert "counterpart" not in d["directories"]


def test_existing_config_without_counterpart_loads():
    """Existing config with custom stacks loads without counterpart error."""
    from prompterator.config.schema import Config

    # Simulates an existing config that only has a custom stack, no "default"
    config = Config.model_validate({
        "stacks": {"my-stack": {"runner": "openai"}},
        "author": {"stack": "my-stack"},
        "editor": {"stack": "my-stack"},
        "critic": {"stack": "my-stack"},
    })
    # Should not raise — counterpart is optional and not validated
    assert config.counterpart.stack == "default"


def test_counterpart_directories_config():
    """counterpart directions can be configured in directories."""
    from prompterator.config.schema import Config

    config = Config(directories={"counterpart": "scripts/user.txt"})
    assert config.directories.counterpart == "scripts/user.txt"

    config2 = Config(directories={"counterpart": ["scripts/a.txt", "scripts/b.txt"]})
    assert config2.directories.counterpart == ["scripts/a.txt", "scripts/b.txt"]


# ---------------------------------------------------------------------------
# resolve_counterpart
# ---------------------------------------------------------------------------

def test_resolve_counterpart_from_config(tmp_path):
    """Resolve counterpart directions from config."""
    from prompterator.commands.resolve import resolve_counterpart
    from prompterator.config.schema import Config

    directions = tmp_path / "user_directions.txt"
    directions.write_text("You are a curious user asking about code.")

    config = Config(directories={"counterpart": str(directions)})
    result = resolve_counterpart(config, tmp_path)

    assert len(result) == 1
    assert "curious user" in result[0]


def test_resolve_counterpart_cli_override(tmp_path):
    """CLI flag overrides config."""
    from prompterator.commands.resolve import resolve_counterpart
    from prompterator.config.schema import Config

    cli_file = tmp_path / "cli_directions.txt"
    cli_file.write_text("CLI directions")

    config = Config()
    result = resolve_counterpart(config, tmp_path, cli_counterpart=cli_file)

    assert len(result) == 1
    assert result[0] == "CLI directions"


def test_resolve_counterpart_empty(tmp_path):
    """No counterpart configured returns empty list."""
    from prompterator.commands.resolve import resolve_counterpart
    from prompterator.config.schema import Config

    config = Config()
    result = resolve_counterpart(config, tmp_path)
    assert result == []


def test_resolve_counterpart_multiple(tmp_path):
    """Multiple counterpart directions files."""
    from prompterator.commands.resolve import resolve_counterpart
    from prompterator.config.schema import Config

    d1 = tmp_path / "user1.txt"
    d1.write_text("Beginner user")
    d2 = tmp_path / "user2.txt"
    d2.write_text("Expert user")

    config = Config(directories={"counterpart": [str(d1), str(d2)]})
    result = resolve_counterpart(config, tmp_path)

    assert len(result) == 2
    assert "Beginner" in result[0]
    assert "Expert" in result[1]
