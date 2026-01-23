"""Prompt improvement logic - generate improved prompts via LLM."""

from pathlib import Path

from prompterator.models.issue import IssueFile
from prompterator.runners.llm import LLMClient


def _build_improvement_prompt(
    original_prompt: str,
    issue_file: IssueFile,
) -> str:
    """Build the prompt for generating improvements."""
    issues_text = []
    for issue in issue_file.issues:
        issues_text.append(f"- [{issue.severity.upper()}] {issue.category}: {issue.summary}")

    issues_section = "\n".join(issues_text) if issues_text else "No specific issues identified."

    return f"""You are an expert prompt engineer. Improve the following prompt to address the identified issues.

ORIGINAL PROMPT:
---
{original_prompt}
---

ISSUES TO ADDRESS:
{issues_section}

INSTRUCTIONS:
1. Address each issue while maintaining the original intent
2. Improve clarity and specificity where noted
3. Add examples if completeness is an issue
4. Fix any accuracy or tone problems
5. Maintain consistent formatting

Provide the improved prompt only, without explanations or commentary.

IMPROVED PROMPT:"""


def generate_improved_prompt(
    prompt_path: Path,
    issue_file: IssueFile,
    llm_client: LLMClient,
) -> str:
    """Generate an improved version of a prompt.

    Args:
        prompt_path: Path to the original prompt file.
        issue_file: IssueFile with issues to address.
        llm_client: LLM client for generation.

    Returns:
        Improved prompt text.
    """
    with open(prompt_path) as f:
        original_prompt = f.read()

    improvement_prompt = _build_improvement_prompt(original_prompt, issue_file)

    system = (
        "You are an expert prompt engineer specializing in improving prompts "
        "for clarity, completeness, and effectiveness. Output only the improved "
        "prompt text without any additional commentary."
    )

    improved = llm_client.generate(
        improvement_prompt,
        system=system,
        temperature=0.7,
    )

    return improved.strip()


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
