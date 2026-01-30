"""Iteration models for tuning loop records."""

from pydantic import BaseModel, Field

from prompterator.models.result import EvalResult, ResultSummary


class PromptDiff(BaseModel):
    """Before/after text of a prompt change."""

    before: str = Field(description="Prompt text before this iteration")
    after: str = Field(description="Prompt text after this iteration")


class IterationRecord(BaseModel):
    """Record of a single tuning iteration."""

    iteration: int = Field(description="Iteration number (1-based)")
    prompt_text: str = Field(description="Prompt text after improvement")
    rationale: str = Field(description="LLM rationale for the change")
    diff: PromptDiff = Field(description="Before/after prompt text")
    eval_results: list[EvalResult] = Field(default_factory=list, description="Eval results")
    summary: ResultSummary = Field(description="Summary of eval results")
    metric_deltas: dict[str, float] = Field(
        default_factory=dict, description="Per-eval score deltas vs previous iteration"
    )
    l2_output: str | None = Field(default=None, description="Raw LLM response from improvement")


class TuneReport(BaseModel):
    """Complete tuning loop report."""

    version: str = Field(default="1.0", description="Report format version")
    prompt_ref: str = Field(description="Reference to the original prompt")
    max_iterations: int = Field(description="Maximum iterations configured")
    iterations: list[IterationRecord] = Field(default_factory=list, description="Iteration records")
    final_prompt: str = Field(description="Final prompt text")
    final_summary: ResultSummary = Field(description="Final eval summary")
    metric_table: list[dict] = Field(
        default_factory=list,
        description="Per-eval metric history [{eval_id, before, after, delta}]",
    )

    def to_yaml_dict(self) -> dict:
        """Convert to dictionary suitable for YAML serialization."""
        return {
            "version": self.version,
            "prompt_ref": self.prompt_ref,
            "max_iterations": self.max_iterations,
            "iterations": [
                {
                    "iteration": rec.iteration,
                    "rationale": rec.rationale,
                    "diff": {"before_length": len(rec.diff.before), "after_length": len(rec.diff.after)},
                    "summary": {
                        "overall_score": rec.summary.overall_score,
                        "verdict": rec.summary.verdict,
                        "passed_count": rec.summary.passed_count,
                        "failed_count": rec.summary.failed_count,
                    },
                    "metric_deltas": rec.metric_deltas,
                }
                for rec in self.iterations
            ],
            "final_prompt": self.final_prompt,
            "final_summary": {
                "overall_score": self.final_summary.overall_score,
                "verdict": self.final_summary.verdict,
                "passed_count": self.final_summary.passed_count,
                "failed_count": self.final_summary.failed_count,
            },
            "metric_table": self.metric_table,
        }
