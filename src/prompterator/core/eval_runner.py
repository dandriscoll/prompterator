"""Eval execution logic - run evals against prompts."""

from pathlib import Path

from prompterator.models.eval import Eval, EvalFile
from prompterator.models.result import EvalResult, ResultFile, ResultSummary
from prompterator.runners.llm import LLMClient


def _build_rubric_prompt(prompt_content: str, criteria: list[str]) -> str:
    """Build a prompt for rubric evaluation."""
    criteria_list = "\n".join(f"- {c}" for c in criteria)
    return f"""Evaluate the following prompt against these criteria:

{criteria_list}

For each criterion, respond with PASS or FAIL and a brief explanation.

PROMPT TO EVALUATE:
---
{prompt_content}
---

Respond in this format:
CRITERION: [criterion text]
RESULT: PASS or FAIL
REASON: [brief explanation]

After evaluating all criteria, provide:
OVERALL: PASS (if all criteria pass) or FAIL (if any fail)
SCORE: [0.0-1.0 based on pass rate]"""


def _parse_rubric_response(response: str, criteria: list[str]) -> tuple[bool, float, str]:
    """Parse the LLM response for rubric evaluation.

    Returns:
        Tuple of (passed, score, details).
    """
    lines = response.strip().split("\n")

    passed_count = 0
    total_count = len(criteria)
    details = []

    for line in lines:
        line = line.strip()
        if line.startswith("RESULT:"):
            result = line.split(":", 1)[1].strip().upper()
            if result == "PASS":
                passed_count += 1
        elif line.startswith("OVERALL:"):
            pass  # We calculate our own
        elif line.startswith("REASON:"):
            details.append(line.split(":", 1)[1].strip())

    if total_count == 0:
        score = 1.0
    else:
        score = passed_count / total_count

    passed = passed_count == total_count
    return passed, score, "; ".join(details[:3])


def run_eval(
    eval_spec: Eval,
    prompt_content: str,
    llm_client: LLMClient,
) -> EvalResult:
    """Run a single evaluation against a prompt.

    Args:
        eval_spec: The eval specification to run.
        prompt_content: Content of the prompt to evaluate.
        llm_client: LLM client for running evaluations.

    Returns:
        EvalResult with pass/fail and score.
    """
    if eval_spec.type == "rubric" and eval_spec.rubric:
        criteria = eval_spec.rubric.criteria
        eval_prompt = _build_rubric_prompt(prompt_content, criteria)

        system = "You are an expert prompt evaluator. Be objective and thorough."

        response = llm_client.generate(eval_prompt, system=system, temperature=0.3)
        passed, score, details = _parse_rubric_response(response, criteria)

        return EvalResult(
            eval_id=eval_spec.id,
            passed=passed,
            score=score,
            details=details,
        )

    elif eval_spec.type == "assertion" and eval_spec.assertion:
        # Simple assertion check
        assertion_prompt = f"""Check if this prompt satisfies the following assertion:

ASSERTION: {eval_spec.assertion}

PROMPT:
---
{prompt_content}
---

Respond with only:
RESULT: PASS or FAIL
REASON: [brief explanation]"""

        system = "You are an expert prompt evaluator. Be objective."
        response = llm_client.generate(assertion_prompt, system=system, temperature=0.3)

        passed = "RESULT: PASS" in response.upper()
        return EvalResult(
            eval_id=eval_spec.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            details=response,
        )

    else:
        # Unknown or unsupported eval type
        return EvalResult(
            eval_id=eval_spec.id,
            passed=False,
            score=0.0,
            details=f"Unsupported eval type: {eval_spec.type}",
        )


def run_all_evals(
    eval_file: EvalFile,
    prompt_path: Path,
    llm_client: LLMClient,
) -> ResultFile:
    """Run all evaluations against a prompt.

    Args:
        eval_file: EvalFile containing all eval specs.
        prompt_path: Path to the prompt file to evaluate.
        llm_client: LLM client for running evaluations.

    Returns:
        ResultFile with all results and summary.
    """
    with open(prompt_path) as f:
        prompt_content = f.read()

    results = []
    for eval_spec in eval_file.evals:
        result = run_eval(eval_spec, prompt_content, llm_client)
        results.append(result)

    # Calculate summary
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    if results:
        overall_score = sum(r.score for r in results) / len(results)
    else:
        overall_score = 1.0

    if passed_count == len(results):
        verdict = "PASS"
    elif passed_count == 0:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"

    summary = ResultSummary(
        overall_score=overall_score,
        verdict=verdict,
        passed_count=passed_count,
        failed_count=failed_count,
    )

    return ResultFile(
        version="1.0",
        prompt_tested=str(prompt_path),
        results=results,
        summary=summary,
    )


def save_result_file(result_file: ResultFile, path: Path) -> None:
    """Save a result file to disk."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(result_file.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)
