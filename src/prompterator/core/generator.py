"""Generator - run a prompt through the Author LLM and capture output."""

from pathlib import Path

from prompterator.runners.llm import LLMClient


def generate_from_prompt(
    prompt_path: Path,
    llm: LLMClient,
    *,
    system: str | None = None,
    content: str | None = None,
    timeout: int = 300,
    output_path: str | None = None,
) -> str:
    """Send a prompt file through the Author LLM and return the response.

    When content is provided:
    - If the prompt contains ``{{INPUT}}``, content is substituted in and
      the result is sent as the user message.
    - Otherwise, the prompt is sent as the system message and content is
      sent as the user message.

    Args:
        prompt_path: Path to the prompt file.
        llm: Configured LLMClient for the author role.
        system: Optional system prompt override.
        content: Optional content to pair with the prompt.
        timeout: LLM call timeout in seconds.

    Returns:
        The LLM-generated text.
    """
    prompt_text = prompt_path.read_text()

    if content is not None:
        if "{{INPUT}}" in prompt_text:
            user_message = prompt_text.replace("{{INPUT}}", content)
            return llm.generate(user_message, system=system, timeout=timeout,
                                output_path=output_path)
        else:
            # Prompt becomes system, content becomes user message
            effective_system = prompt_text
            if system:
                effective_system = system + "\n\n" + prompt_text
            return llm.generate(content, system=effective_system, timeout=timeout,
                                output_path=output_path)

    return llm.generate(prompt_text, system=system, timeout=timeout,
                        output_path=output_path)
