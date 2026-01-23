"""Eval models for evaluation specifications."""

from typing import Literal

from pydantic import BaseModel, Field


class EvalRubric(BaseModel):
    """Rubric-based evaluation criteria."""

    criteria: list[str] = Field(description="List of criteria to evaluate")
    scoring: Literal["all_required", "any_required", "weighted"] = Field(
        default="all_required",
        description="Scoring method",
    )
    weights: list[float] | None = Field(
        default=None,
        description="Weights for weighted scoring",
    )


class Eval(BaseModel):
    """A single evaluation specification."""

    id: str = Field(description="Unique eval identifier")
    type: Literal["rubric", "comparison", "assertion"] = Field(
        default="rubric",
        description="Type of evaluation",
    )
    issue_ref: str | None = Field(default=None, description="Reference to source issue")
    rubric: EvalRubric | None = Field(default=None, description="Rubric for rubric-type evals")
    assertion: str | None = Field(default=None, description="Assertion for assertion-type evals")
    description: str | None = Field(default=None, description="Human-readable description")


class EvalFile(BaseModel):
    """Complete eval file structure."""

    version: str = Field(default="1.0", description="File format version")
    prompt_ref: str = Field(description="Reference to the prompt file")
    evals: list[Eval] = Field(default_factory=list, description="List of evaluations")

    def to_yaml_dict(self) -> dict:
        """Convert to dictionary suitable for YAML serialization."""
        result = {
            "version": self.version,
            "prompt_ref": self.prompt_ref,
            "evals": [],
        }
        for eval_item in self.evals:
            eval_dict: dict = {
                "id": eval_item.id,
                "type": eval_item.type,
            }
            if eval_item.issue_ref:
                eval_dict["issue_ref"] = eval_item.issue_ref
            if eval_item.description:
                eval_dict["description"] = eval_item.description
            if eval_item.rubric:
                eval_dict["rubric"] = {
                    "criteria": eval_item.rubric.criteria,
                    "scoring": eval_item.rubric.scoring,
                }
                if eval_item.rubric.weights:
                    eval_dict["rubric"]["weights"] = eval_item.rubric.weights
            if eval_item.assertion:
                eval_dict["assertion"] = eval_item.assertion
            result["evals"].append(eval_dict)
        return result
