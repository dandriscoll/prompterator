"""Subprocess wrapper for critic script mode."""

import shutil
import subprocess
from pathlib import Path

import yaml

from prompterator.models.eval import Eval
from prompterator.models.result import EvalResult


class CriticScriptError(Exception):
    """Error from critic script."""

    pass


def _find_script(script: str) -> str:
    """Resolve critic script path."""
    path = Path(script)
    if path.exists():
        return str(path)

    found = shutil.which(script)
    if found:
        return found

    raise CriticScriptError(f"Critic script not found: {script}")


def _build_script_input(prompt_content: str, eval_spec: Eval) -> str:
    """Build YAML input for the critic script."""
    data: dict = {
        "prompt": prompt_content,
        "eval": {
            "id": eval_spec.id,
            "type": eval_spec.type,
        },
    }
    if eval_spec.rubric:
        rubric: dict = {
            "criteria": eval_spec.rubric.criteria,
            "scoring": eval_spec.rubric.scoring,
        }
        if eval_spec.rubric.weights:
            rubric["weights"] = eval_spec.rubric.weights
        data["eval"]["rubric"] = rubric
    if eval_spec.assertion:
        data["eval"]["assertion"] = eval_spec.assertion
    if eval_spec.description:
        data["eval"]["description"] = eval_spec.description
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _parse_script_output(stdout: str, eval_id: str) -> EvalResult:
    """Parse YAML output from the critic script."""
    try:
        data = yaml.safe_load(stdout)
    except yaml.YAMLError as e:
        raise CriticScriptError(f"Invalid YAML output from critic script: {e}")

    if not isinstance(data, dict):
        raise CriticScriptError(f"Critic script output must be a YAML mapping, got {type(data).__name__}")

    return EvalResult(
        eval_id=data.get("eval_id", eval_id),
        passed=bool(data.get("passed", False)),
        score=float(data.get("score", 0.0)),
        details=data.get("details"),
    )


def run_script_eval(
    script: str,
    eval_spec: Eval,
    prompt_content: str,
    timeout: int = 60,
) -> EvalResult:
    """Run a single eval via an external critic script.

    Args:
        script: Path or name of the critic script.
        eval_spec: The eval specification to run.
        prompt_content: Content of the prompt to evaluate.
        timeout: Timeout in seconds.

    Returns:
        EvalResult from the script.
    """
    executable = _find_script(script)
    stdin_data = _build_script_input(prompt_content, eval_spec)

    try:
        result = subprocess.run(
            [executable],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return _parse_script_output(result.stdout, eval_spec.id)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown error"
        raise CriticScriptError(f"Critic script failed: {error_msg}")
    except subprocess.TimeoutExpired:
        raise CriticScriptError(f"Critic script timed out after {timeout}s")
    except FileNotFoundError:
        raise CriticScriptError(f"Critic script not found: {executable}")
