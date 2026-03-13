"""Generator - run a prompt through the Author LLM and capture output."""

from pathlib import Path

from prompterator.runners.llm import LLMClient


def generate_from_prompt(
    prompt_path: Path,
    llm: LLMClient,
    *,
    system: str | None = None,
    timeout: int = 300,
) -> str:
    """Send a prompt file through the Author LLM and return the response.

    Args:
        prompt_path: Path to the prompt file.
        llm: Configured LLMClient for the author role.
        system: Optional system prompt override.
        timeout: LLM call timeout in seconds.

    Returns:
        The LLM-generated text.
    """
    prompt_text = prompt_path.read_text()
    return llm.generate(prompt_text, system=system, timeout=timeout)
