"""Eval execution logic - run evals against prompts."""

from __future__ import annotations

from pathlib import Path

from prompterator.models.eval import Eval, EvalFile
from prompterator.models.result import EvalResult, ResultFile, ResultSummary
from prompterator.runners.llm import LLMClient


def _build_rubric_prompt(
    output_content: str,
    criteria: list[str],
    *,
    is_prompt_eval: bool = False,
) -> str:
    """Build a prompt for rubric evaluation."""
    criteria_blocks = []
    for i, c in enumerate(criteria, 1):
        criteria_blocks.append(f"CRITERION {i}: {c}")
    criteria_list = "\n".join(criteria_blocks)

    if is_prompt_eval:
        context = (
            "You are evaluating a PROMPT TEMPLATE — text that will be sent "
            "to an LLM to control its behavior. The criteria ask whether the "
            "prompt contains specific INSTRUCTIONS or RULES. A criterion like "
            '"Prompt prohibits X" means the prompt must contain an instruction '
            "that tells the LLM not to do X. Evaluate whether such instructions "
            "exist in the prompt, not whether the prompt itself does X."
        )
        label = "PROMPT TEMPLATE TO EVALUATE"
    else:
        context = (
            "You are evaluating an LLM's output against specific criteria. "
            "Read the output carefully, then evaluate ONLY the numbered "
            "criteria below."
        )
        label = "OUTPUT TO EVALUATE"

    return f"""{context}

{criteria_list}

{label}:
<<<
{output_content}
>>>

For each numbered criterion above, give your verdict:
CRITERION 1: PASS or FAIL
REASON 1: [brief explanation]
(repeat for each criterion)

OVERALL: PASS if ALL criteria pass, FAIL if ANY criterion fails"""


def _parse_rubric_response(response: str, criteria: list[str]) -> tuple[bool, float, str]:
    """Parse the LLM response for rubric evaluation.

    Returns:
        Tuple of (passed, score, details).
    """
    import re
    lines = response.strip().split("\n")

    passed_count = 0
    total_count = len(criteria)
    details = []

    for line in lines:
        line = line.strip()
        # Match "CRITERION N: PASS/FAIL" or "RESULT: PASS/FAIL" patterns
        criterion_match = re.match(r'^CRITERION\s*\d*\s*:\s*(PASS|FAIL)', line, re.IGNORECASE)
        result_match = re.match(r'^RESULT\s*\d*\s*:\s*(PASS|FAIL)', line, re.IGNORECASE)
        match = criterion_match or result_match
        if match:
            if match.group(1).upper() == "PASS":
                passed_count += 1
        elif line.startswith("OVERALL:"):
            pass  # We calculate our own
        elif re.match(r'^REASON\s*\d*\s*:', line):
            details.append(re.split(r'^REASON\s*\d*\s*:\s*', line, 1)[-1].strip())

    if total_count == 0:
        score = 1.0
    else:
        score = passed_count / total_count

    passed = passed_count == total_count
    return passed, score, "; ".join(details[:3])


def run_eval(
    eval_spec: Eval,
    output_content: str,
    llm_client: LLMClient | None = None,
    *,
    prompt_content: str | None = None,
    script: str | None = None,
    script_timeout: int = 60,
) -> EvalResult:
    """Run a single evaluation against LLM-generated output.

    Args:
        eval_spec: The eval specification to run.
        output_content: The LLM-generated output to evaluate.
        llm_client: LLM client for running evaluations (used in llm mode).
        script: Path to critic script (used in script mode).
        script_timeout: Timeout for script execution in seconds.

    Returns:
        EvalResult with pass/fail and score.
    """
    # Script mode: delegate entirely to the external script
    if script is not None:
        from prompterator.runners.critic_script import run_script_eval

        return run_script_eval(script, eval_spec, output_content, timeout=script_timeout)

    if eval_spec.type == "rubric" and eval_spec.rubric:
        criteria = eval_spec.rubric.criteria
        # If criteria reference the prompt itself (e.g. "Prompt prohibits..."),
        # evaluate the prompt text directly, not the author's output
        is_prompt_eval = prompt_content and any(
            c.strip().lower().startswith("prompt ") for c in criteria
        )
        content_to_eval = prompt_content if is_prompt_eval else output_content
        eval_prompt = _build_rubric_prompt(
            content_to_eval, criteria, is_prompt_eval=is_prompt_eval,
        )

        system = "You are an expert evaluator. Be objective and thorough."

        response = llm_client.generate(eval_prompt, system=system, temperature=0.3)
        passed, score, details = _parse_rubric_response(response, criteria)

        return EvalResult(
            eval_id=eval_spec.id,
            passed=passed,
            score=score,
            details=details,
        )

    elif eval_spec.type == "assertion" and eval_spec.assertion:
        assertion_prompt = f"""Check if this output satisfies the following assertion:

ASSERTION: {eval_spec.assertion}

OUTPUT:
---
{output_content}
---

Respond with only:
RESULT: PASS or FAIL
REASON: [brief explanation]"""

        system = "You are an expert output evaluator. Be objective."
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
    llm_client: LLMClient | None = None,
    *,
    author_llm: LLMClient | None = None,
    samples: int = 1,
    confidence_threshold: float = 0.90,
    script: str | None = None,
    script_timeout: int = 60,
) -> ResultFile:
    """Generate output from a prompt and evaluate it.

    When samples > 1, generates and evaluates multiple outputs. Each eval
    passes only if the fraction of passing samples meets the confidence
    threshold.

    Args:
        eval_file: EvalFile containing all eval specs.
        prompt_path: Path to the prompt file.
        llm_client: LLM client for running evaluations (critic).
        author_llm: LLM client for generating output from the prompt.
        samples: Number of generate→eval samples to run.
        confidence_threshold: Fraction of samples that must pass per eval.
        script: Path to critic script (script mode).
        script_timeout: Timeout for script execution in seconds.

    Returns:
        ResultFile with aggregated results, summary, and last generated output.
    """
    with open(prompt_path) as f:
        prompt_content = f.read()

    # Collect per-eval results across samples
    # eval_id -> list of EvalResult
    per_eval: dict[str, list[EvalResult]] = {
        spec.id: [] for spec in eval_file.evals
    }
    last_output = None

    for _ in range(samples):
        # Generate output from the prompt
        if author_llm is not None:
            output_content = author_llm.generate(prompt_content)
        else:
            output_content = prompt_content
        last_output = output_content

        for eval_spec in eval_file.evals:
            result = run_eval(
                eval_spec, output_content, llm_client,
                prompt_content=prompt_content,
                script=script, script_timeout=script_timeout,
            )
            per_eval[eval_spec.id].append(result)

    # Aggregate: each eval passes if pass_rate >= confidence_threshold
    results = []
    for eval_spec in eval_file.evals:
        sample_results = per_eval[eval_spec.id]
        pass_rate = sum(1 for r in sample_results if r.passed) / len(sample_results)
        passed = pass_rate >= confidence_threshold
        # Score reflects the pass rate so it's consistent with the verdict
        avg_score = pass_rate

        # Collect details from failing samples
        fail_details = [r.details for r in sample_results if not r.passed and r.details]
        if samples > 1:
            details = f"pass_rate={pass_rate:.0%} ({sum(1 for r in sample_results if r.passed)}/{len(sample_results)})"
            if fail_details:
                details += f"; {fail_details[0]}"
        else:
            details = sample_results[0].details

        results.append(EvalResult(
            eval_id=eval_spec.id,
            passed=passed,
            score=avg_score,
            details=details,
        ))

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
        generated_output=last_output if author_llm is not None else None,
        results=results,
        summary=summary,
    )


def load_result_file(path: Path) -> ResultFile:
    """Load a result file from disk."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    results = [
        EvalResult(
            eval_id=r["eval_id"],
            passed=r["passed"],
            score=r["score"],
            details=r.get("details"),
        )
        for r in data.get("results", [])
    ]
    summary_data = data.get("summary", {})
    summary = ResultSummary(
        overall_score=summary_data.get("overall_score", 0.0),
        verdict=summary_data.get("verdict", "FAIL"),
        passed_count=summary_data.get("passed_count", 0),
        failed_count=summary_data.get("failed_count", 0),
    )
    return ResultFile(
        version=data.get("version", "1.0"),
        prompt_tested=data.get("prompt_tested", ""),
        generated_output=data.get("generated_output"),
        results=results,
        summary=summary,
    )


def find_latest_results(results_dir: Path, base_name: str) -> Path | None:
    """Find the most recent results file for a given prompt base name.

    Searches timestamped run directories (YYYYMMDD-HHMMSS) in reverse order,
    returning the first matching .results.yaml file found.
    """
    if not results_dir.exists():
        return None

    import re

    run_dir_pattern = re.compile(r"^run\d{3}-\d{8}-\d{6}$")
    run_dirs = sorted(
        (d for d in results_dir.iterdir() if d.is_dir() and run_dir_pattern.match(d.name)),
        reverse=True,
    )

    for run_dir in run_dirs:
        candidates = sorted(run_dir.glob(f"{base_name}.results.yaml"), reverse=True)
        if candidates:
            return candidates[0]
        # Also match tune iteration results (e.g. base_name.001.results.yaml)
        candidates = sorted(run_dir.glob(f"{base_name}.*.results.yaml"), reverse=True)
        if candidates:
            return candidates[0]

    return None


def save_result_file(result_file: ResultFile, path: Path) -> None:
    """Save a result file to disk."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(result_file.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)
