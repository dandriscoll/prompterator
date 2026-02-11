"""Prompt improvement logic - generate improved prompts via LLM."""

import json
from pathlib import Path

from prompterator.models.issue import IssueFile
from prompterator.models.result import EvalResult
from prompterator.runners.llm import LLMClient


def _build_improvement_prompt(
    original_prompt: str,
    issue_file: IssueFile,
    eval_results: list[EvalResult] | None = None,
    iteration: int | None = None,
) -> str:
    """Build the prompt for generating improvements."""
    issues_text = []
    for issue in issue_file.issues:
        issues_text.append(f"- [{issue.severity.upper()}] {issue.category}: {issue.summary}")

    issues_section = "\n".join(issues_text) if issues_text else "No specific issues identified."

    prompt = f"""You are a prompt editor. Your task is to improve the following prompt by addressing ALL of the failing issues listed below. Add explicit constraints and instructions to the prompt so that each issue is prevented.

ORIGINAL PROMPT:
---
{original_prompt}
---

ISSUES TO ADDRESS:
{issues_section}"""

    if eval_results:
        results_text = []
        for r in eval_results:
            status = "PASS" if r.passed else "FAIL"
            results_text.append(f"- [{status}] {r.eval_id}: score={r.score:.2f}" + (f" ({r.details})" if r.details else ""))
        prompt += f"""

PREVIOUS EVAL RESULTS:
{chr(10).join(results_text)}"""

    if iteration is not None:
        prompt += f"""

ITERATION: {iteration}"""

    prompt += """

INSTRUCTIONS:
1. Address ALL failing issues by adding explicit constraints to the prompt
2. For each failing issue, add a clear, direct instruction that prevents the problem
3. Preserve the original intent and overall structure of the prompt
4. Focus on adding constraints rather than rewriting - keep existing text and append rules

Respond with a JSON object (no markdown fencing):
{
  "rationale": "Brief explanation of what you changed and why",
  "changed_section": "Description of which part was modified",
  "improved_prompt": "The full improved prompt text"
}"""

    return prompt


_SURGICAL_SYSTEM = (
    "You are a prompt editor. Address ALL failing issues in a single pass by adding "
    "explicit constraints and instructions to the prompt. "
    "Output valid JSON with rationale, changed_section, and improved_prompt fields. "
    "Preserve existing prompt text and append clear rules that prevent each problem."
)


def _parse_improvement_response(response: str) -> tuple[str, str]:
    """Parse the LLM response to extract rationale and improved prompt.

    Returns:
        Tuple of (rationale, improved_prompt).
    """
    # Try JSON parse first
    try:
        data = json.loads(response)
        rationale = data.get("rationale", "")
        improved = data.get("improved_prompt", "")
        if improved:
            return rationale, improved
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON within the response
    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(response[start:end])
            rationale = data.get("rationale", "")
            improved = data.get("improved_prompt", "")
            if improved:
                return rationale, improved
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: treat entire response as the improved prompt
    return "No structured rationale provided", response.strip()


def generate_improved_prompt(
    prompt_path: Path,
    issue_file: IssueFile,
    llm_client: LLMClient,
    eval_results: list[EvalResult] | None = None,
    iteration: int | None = None,
) -> str:
    """Generate an improved version of a prompt.

    Args:
        prompt_path: Path to the original prompt file.
        issue_file: IssueFile with issues to address.
        llm_client: LLM client for generation.
        eval_results: Previous eval results for context.
        iteration: Current iteration number.

    Returns:
        Improved prompt text.
    """
    with open(prompt_path) as f:
        original_prompt = f.read()

    improvement_prompt = _build_improvement_prompt(
        original_prompt, issue_file, eval_results, iteration
    )

    improved = llm_client.generate(
        improvement_prompt,
        system=_SURGICAL_SYSTEM,
        temperature=0.7,
    )

    _rationale, prompt_text = _parse_improvement_response(improved)
    return prompt_text


def generate_improved_prompt_with_rationale(
    prompt_text: str,
    issue_file: IssueFile,
    llm_client: LLMClient,
    eval_results: list[EvalResult] | None = None,
    iteration: int | None = None,
) -> tuple[str, str, str]:
    """Generate an improved prompt with rationale and raw output.

    Args:
        prompt_text: Current prompt text (not a file path).
        issue_file: IssueFile with issues to address.
        llm_client: LLM client for generation.
        eval_results: Previous eval results for context.
        iteration: Current iteration number.

    Returns:
        Tuple of (improved_prompt, rationale, raw_llm_output).
    """
    improvement_prompt = _build_improvement_prompt(
        prompt_text, issue_file, eval_results, iteration
    )

    raw_response = llm_client.generate(
        improvement_prompt,
        system=_SURGICAL_SYSTEM,
        temperature=0.7,
    )

    rationale, improved = _parse_improvement_response(raw_response)
    return improved, rationale, raw_response


def generate_multiple_variants(
    prompt_path: Path,
    issue_file: IssueFile,
    llm_client: LLMClient,
    num_variants: int = 3,
) -> list[str]:
    """Generate multiple improved variants of a prompt.

    Args:
        prompt_path: Path to the original prompt file.
        issue_file: IssueFile with issues to address.
        llm_client: LLM client for generation.
        num_variants: Number of variants to generate.

    Returns:
        List of improved prompt texts.
    """
    variants = []
    for i in range(num_variants):
        # Use slightly different temperatures for variety
        temp = 0.6 + (i * 0.15)
        improved = generate_improved_prompt(prompt_path, issue_file, llm_client)
        variants.append(improved)

    return variants


def save_improved_prompt(content: str, path: Path) -> None:
    """Save an improved prompt to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
