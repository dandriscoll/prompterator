from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config
from .examples import EvaluationResult, get_backend
from .file_modes import resolve_file_mode, write_output
from .llm import invoke_llm


@dataclass(frozen=True)
class WorkflowResult:
    refined_prompt: str
    operator_output: str
    evaluation: EvaluationResult
    applied_path: Optional[Path]
    file_mode: Optional[str]


def run_workflow(
    *,
    config: Config,
    initial_prompt: str,
    examples_path: Path,
    feedback: str,
    apply_to: Optional[Path],
    cwd: Path,
) -> WorkflowResult:
    backend = get_backend(config.examples_module)
    examples = backend.load(examples_path)
    examples_text = backend.format_for_editor(examples)

    editor_prompt = build_editor_prompt(
        initial_prompt=initial_prompt,
        examples_text=examples_text,
        feedback=feedback,
    )
    refined_prompt = invoke_llm(
        config.editor_endpoint,
        editor_prompt,
        config.editor_api_key,
        config.request_timeout,
    )

    operator_output = invoke_llm(
        config.operator_endpoint,
        refined_prompt,
        config.operator_api_key,
        config.request_timeout,
    )

    evaluation = backend.evaluate(examples, operator_output)

    applied_path: Optional[Path] = None
    file_mode: Optional[str] = None
    if apply_to:
        file_mode = resolve_file_mode(config.file_mode, cwd)
        applied_path = write_output(
            apply_to,
            operator_output,
            file_mode,
            config.output_suffix,
        )

    return WorkflowResult(
        refined_prompt=refined_prompt,
        operator_output=operator_output,
        evaluation=evaluation,
        applied_path=applied_path,
        file_mode=file_mode,
    )


def build_editor_prompt(*, initial_prompt: str, examples_text: str, feedback: str) -> str:
    prompt = initial_prompt.strip()
    feedback_text = feedback.strip() or "None"
    if not prompt:
        prompt = "<empty>"
    return (
        "You are the Editor LLM. Improve the input prompt using the examples and feedback.\n"
        "Return only the improved prompt.\n\n"
        f"Examples:\n{examples_text}\n\n"
        f"Feedback:\n{feedback_text}\n\n"
        f"Original prompt:\n{prompt}\n"
    )
