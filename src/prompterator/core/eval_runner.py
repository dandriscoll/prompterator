"""Eval execution logic - run evals against prompts."""

from __future__ import annotations

from pathlib import Path

from prompterator.models.eval import Eval, EvalFile
from prompterator.models.feedback import Feedback
from prompterator.models.issue import IssueFile
from prompterator.models.result import EvalResult, ResultFile, ResultSummary
from prompterator.runners.llm import LLMClient


def _build_rubric_prompt(
    output_content: str,
    criteria: list[str],
    *,
    is_prompt_eval: bool = False,
    input_content: str | None = None,
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

    input_section = ""
    if input_content and not is_prompt_eval:
        input_section = f"""

ORIGINAL INPUT (the content the LLM was given to work with):
<<<
{input_content}
>>>
"""

    return f"""{context}

{criteria_list}
{input_section}
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
    input_content: str | None = None,
    script: str | None = None,
    script_timeout: int = 60,
) -> EvalResult:
    """Run a single evaluation against LLM-generated output.

    Args:
        eval_spec: The eval specification to run.
        output_content: The LLM-generated output to evaluate.
        llm_client: LLM client for running evaluations (used in llm mode).
        input_content: The original content/input given to the author LLM.
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
            input_content=input_content,
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
        input_section = ""
        if input_content:
            input_section = f"""
ORIGINAL INPUT:
---
{input_content}
---

"""
        assertion_prompt = f"""Check if this output satisfies the following assertion:

ASSERTION: {eval_spec.assertion}
{input_section}
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


def _prepare_author_input(prompt_content: str, content_text: str | None) -> tuple[str, str | None]:
    """Prepare the author LLM input from prompt and content.

    Returns:
        Tuple of (user_message, system_message).
        If content has {{INPUT}}, substitution is done and system is None.
        If content is provided without {{INPUT}}, prompt becomes system.
        If no content, prompt is the user message with no system.
    """
    if content_text is None:
        return prompt_content, None

    if "{{INPUT}}" in prompt_content:
        return prompt_content.replace("{{INPUT}}", content_text), None

    # Prompt becomes system, content becomes user
    return content_text, prompt_content


def run_multi_turn_dialog(
    prompt_content: str,
    author_llm: LLMClient,
    counterpart_llm: LLMClient,
    counterpart_directions: str,
    *,
    content_text: str | None = None,
    turns: int = 3,
) -> str:
    """Run a multi-turn dialog between the author and counterpart LLMs.

    The author generates from the prompt (with optional content).
    The counterpart replies according to its directions.
    They alternate for the given number of turns.

    Args:
        prompt_content: The author's system prompt.
        author_llm: LLM client for the author role.
        counterpart_llm: LLM client for the counterpart role.
        counterpart_directions: System prompt/directions for the counterpart.
        content_text: Optional initial content (first user message to author).
        turns: Number of author turns (counterpart gets turns-1).

    Returns:
        Full conversation transcript as a single string.
    """
    user_msg, system_msg = _prepare_author_input(prompt_content, content_text)

    # Conversation history for building context
    transcript_lines: list[str] = []
    # Running context strings for each participant
    author_history: list[str] = []
    counterpart_history: list[str] = []

    for turn in range(turns):
        # --- Author turn ---
        if turn == 0:
            # First turn: use prepared input
            author_input = user_msg
            author_system = system_msg
        else:
            # Subsequent turns: counterpart's last message is the user input
            author_input = counterpart_history[-1]
            author_system = prompt_content

        author_response = author_llm.generate(
            _build_conversation_prompt(author_input, author_history),
            system=author_system,
        )
        author_history.append(author_response)
        transcript_lines.append(f"[Author turn {turn + 1}]\n{author_response}")

        # --- Counterpart turn (except after last author turn) ---
        if turn < turns - 1:
            counterpart_input = author_response
            counterpart_response = counterpart_llm.generate(
                _build_conversation_prompt(counterpart_input, counterpart_history),
                system=counterpart_directions,
            )
            counterpart_history.append(counterpart_response)
            transcript_lines.append(f"[Counterpart turn {turn + 1}]\n{counterpart_response}")

    return "\n\n".join(transcript_lines)


def _build_conversation_prompt(
    current_message: str,
    history: list[str],
) -> str:
    """Build a conversation prompt that includes prior history.

    If there is no history, returns the current message as-is.
    With history, prepends prior exchanges so the LLM has context.
    """
    if not history:
        return current_message

    parts = []
    for i, msg in enumerate(history):
        parts.append(f"[Previous response {i + 1}]\n{msg}")
    parts.append(f"[Current message to respond to]\n{current_message}")
    return "\n\n".join(parts)


def map_content_to_evals(
    content_paths: list[Path],
    feedback_list: list[Feedback],
    issue_file: IssueFile,
    eval_file: EvalFile,
) -> dict[int, list[str]] | None:
    """Map each content file to the eval IDs that apply to it.

    Traces the chain: content → feedback (via prior_ref) → issues
    (via evidence source) → evals (via issue_ref).

    Returns:
        Dict mapping content index to list of eval IDs, or None if
        any content file has no feedback coverage (meaning we cannot
        safely filter and must run all evals).
    """
    # Step 1: Map content basename → set of .mb source basenames
    content_to_mb: dict[str, set[str]] = {}
    for fb in feedback_list:
        mb_name = Path(fb.source_file).name
        for entry in fb.entries:
            if entry.prior_ref:
                prior_name = Path(entry.prior_ref).name
                content_to_mb.setdefault(prior_name, set()).add(mb_name)

    # Step 2: Map .mb source basename → set of issue IDs
    mb_to_issues: dict[str, set[str]] = {}
    for issue in issue_file.issues:
        for ev in issue.evidence:
            src_name = Path(ev.source).name
            mb_to_issues.setdefault(src_name, set()).add(issue.id)

    # Step 3: Map issue ID → set of eval IDs
    issue_to_evals: dict[str, set[str]] = {}
    for eval_spec in eval_file.evals:
        if eval_spec.issue_ref:
            issue_to_evals.setdefault(eval_spec.issue_ref, set()).add(eval_spec.id)

    # Step 4: For each content, trace the chain
    result: dict[int, list[str]] = {}
    for i, content_path in enumerate(content_paths):
        content_name = content_path.name
        mb_names = content_to_mb.get(content_name, set())

        if not mb_names:
            # This content has no feedback — can't optimize
            return None

        eval_ids: set[str] = set()
        for mb_name in mb_names:
            for issue_id in mb_to_issues.get(mb_name, set()):
                eval_ids.update(issue_to_evals.get(issue_id, set()))

        result[i] = sorted(eval_ids)

    return result


def run_all_evals(
    eval_file: EvalFile,
    prompt_path: Path,
    llm_client: LLMClient | None = None,
    *,
    author_llm: LLMClient | None = None,
    content_texts: list[str | None] | None = None,
    samples: int = 1,
    ensemble: int = 5,
    confidence_threshold: float = 0.90,
    script: str | None = None,
    script_timeout: int = 60,
    content_eval_map: dict[int, list[str]] | None = None,
    counterpart_llm: LLMClient | None = None,
    counterpart_directions: list[str] | None = None,
    dialog_turns: int = 3,
) -> ResultFile:
    """Generate output from a prompt and evaluate it with ensemble scoring.

    For each (content_file x sample), generates one author output, then
    evaluates it ``ensemble`` times per eval. The per-eval score is the
    mean pass rate across all (outputs x ensemble), normalized to 10.

    When a counterpart LLM and directions are provided, the author output
    is replaced by a multi-turn dialog transcript between author and
    counterpart.

    Args:
        eval_file: EvalFile containing all eval specs.
        prompt_path: Path to the prompt file.
        llm_client: LLM client for running evaluations (critic).
        author_llm: LLM client for generating output from the prompt.
        content_texts: List of content strings to pair with the prompt.
        samples: Number of author outputs per content file.
        ensemble: Number of critic evaluations per output per eval.
        confidence_threshold: Score (out of 10) required for an eval to pass.
        script: Path to critic script (script mode).
        script_timeout: Timeout for script execution in seconds.
        content_eval_map: Optional mapping of content index to eval IDs.
            When provided, only the listed evals run for each content file.
            Evals not listed for any content are still included in results
            but with no data (treated as not run).
        counterpart_llm: LLM client for the counterpart in multi-turn dialog.
        counterpart_directions: List of directions strings for counterpart.
            If provided with counterpart_llm, multi-turn dialog is used
            instead of single-shot author generation. Directions are cycled
            if fewer than content files.
        dialog_turns: Number of author turns in multi-turn dialog (default 3).

    Returns:
        ResultFile with aggregated results, summary, and last generated output.
    """
    with open(prompt_path) as f:
        prompt_content = f.read()

    if content_texts is None:
        content_texts = [None]

    # Collect per-eval ensemble results across all outputs
    # eval_id -> list of pass rates (one per output, each from ensemble votes)
    per_eval_pass_rates: dict[str, list[float]] = {
        spec.id: [] for spec in eval_file.evals
    }
    # Also collect details from failing evaluations
    per_eval_fail_details: dict[str, list[str]] = {
        spec.id: [] for spec in eval_file.evals
    }
    last_output = None
    all_outputs: list[str] = []
    n_outputs = samples * len(content_texts)

    # Determine if multi-turn dialog mode is active
    use_dialog = (
        counterpart_llm is not None
        and counterpart_directions is not None
        and len(counterpart_directions) > 0
    )

    for content_idx, content_text in enumerate(content_texts):
        # Determine which evals to run for this content
        if content_eval_map is not None:
            active_eval_ids = set(content_eval_map.get(content_idx, []))
            active_evals = [e for e in eval_file.evals if e.id in active_eval_ids]
        else:
            active_evals = eval_file.evals

        user_msg, system_msg = _prepare_author_input(prompt_content, content_text)

        for _ in range(samples):
            # Generate one author output (or multi-turn dialog)
            if use_dialog and author_llm is not None:
                # Cycle through directions if fewer than content files
                directions_idx = content_idx % len(counterpart_directions)
                output_content = run_multi_turn_dialog(
                    prompt_content,
                    author_llm,
                    counterpart_llm,
                    counterpart_directions[directions_idx],
                    content_text=content_text,
                    turns=dialog_turns,
                )
            elif author_llm is not None:
                output_content = author_llm.generate(user_msg, system=system_msg)
            else:
                output_content = user_msg
            last_output = output_content
            if author_llm is not None:
                all_outputs.append(output_content)

            # Ensemble evaluate this output
            for eval_spec in active_evals:
                passes = 0
                details_for_output = []
                for _ in range(ensemble):
                    result = run_eval(
                        eval_spec, output_content, llm_client,
                        prompt_content=prompt_content,
                        input_content=content_text,
                        script=script, script_timeout=script_timeout,
                    )
                    if result.passed:
                        passes += 1
                    elif result.details:
                        details_for_output.append(result.details)
                output_pass_rate = passes / ensemble
                per_eval_pass_rates[eval_spec.id].append(output_pass_rate)
                if details_for_output:
                    per_eval_fail_details[eval_spec.id].append(details_for_output[0])

    # Aggregate: score is mean pass rate normalized to 10
    results = []
    skipped_evals = []
    for eval_spec in eval_file.evals:
        rates = per_eval_pass_rates[eval_spec.id]

        if not rates and content_eval_map is not None:
            # Eval was filtered out for all content — skip from results
            skipped_evals.append(eval_spec.id)
            continue

        mean_rate = sum(rates) / len(rates) if rates else 0.0
        score_10 = round(mean_rate * 10, 2)
        passed = score_10 >= confidence_threshold

        # Build details string
        fail_details = per_eval_fail_details[eval_spec.id]
        n_actual_outputs = len(rates)
        if n_actual_outputs > 1 or ensemble > 1:
            total_votes = n_actual_outputs * ensemble
            total_passes = sum(int(r * ensemble) for r in rates)
            details = f"{score_10:.1f}/10 ({total_passes}/{total_votes} votes)"
            if fail_details:
                details += f"; {fail_details[0]}"
        else:
            details = fail_details[0] if fail_details else ""

        results.append(EvalResult(
            eval_id=eval_spec.id,
            passed=passed,
            score=score_10,
            details=details,
        ))

    # Calculate summary — overall score is mean of per-eval scores (already /10)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    if results:
        overall_score = sum(r.score for r in results) / len(results)
    else:
        overall_score = 10.0

    if passed_count == len(results):
        verdict = "PASS"
    elif passed_count == 0 and overall_score <= 2.5:
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
        generated_outputs=all_outputs,
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
