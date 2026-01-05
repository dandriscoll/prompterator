from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    score: Optional[float]
    details: str


class ExamplesBackend:
    def load(self, path: Path) -> Any:
        raise NotImplementedError

    def format_for_editor(self, examples: Any) -> str:
        raise NotImplementedError

    def evaluate(self, examples: Any, output: str) -> EvaluationResult:
        raise NotImplementedError


class RawTextBackend(ExamplesBackend):
    def load(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def format_for_editor(self, examples: str) -> str:
        return examples

    def evaluate(self, examples: str, output: str) -> EvaluationResult:
        return EvaluationResult(
            status="skipped",
            score=None,
            details="Examples backend not available; evaluation skipped.",
        )


class ExternalBackend(ExamplesBackend):
    def __init__(self, module_name: str) -> None:
        self._module = import_module(module_name)
        self._load = _require_callable(self._module, ["load_examples", "load"])
        self._format = _require_callable(
            self._module, ["format_for_editor", "format_examples", "to_editor_text"]
        )
        self._evaluate = _require_callable(
            self._module, ["evaluate_output", "evaluate", "score_output"]
        )

    def load(self, path: Path) -> Any:
        return self._load(str(path))

    def format_for_editor(self, examples: Any) -> str:
        return self._format(examples)

    def evaluate(self, examples: Any, output: str) -> EvaluationResult:
        result = self._evaluate(examples, output)
        return normalize_evaluation(result)


def get_backend(module_name: Optional[str]) -> ExamplesBackend:
    if module_name:
        try:
            return ExternalBackend(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                return RawTextBackend()
            raise
    return RawTextBackend()


def normalize_evaluation(result: Any) -> EvaluationResult:
    if isinstance(result, EvaluationResult):
        return result
    if isinstance(result, dict):
        status = str(result.get("status", "unknown"))
        score = result.get("score")
        details = str(result.get("details", ""))
        return EvaluationResult(status=status, score=score, details=details)
    return EvaluationResult(status="unknown", score=None, details=str(result))


def _require_callable(module: Any, names: list[str]) -> Callable[..., Any]:
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise AttributeError(
        f"Examples module missing required callable (tried: {', '.join(names)})"
    )
