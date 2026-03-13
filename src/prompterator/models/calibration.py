"""Calibration models for eval-label agreement reporting."""

from typing import Literal

from pydantic import BaseModel, Field


class CalibrationExample(BaseModel):
    """Result of calibrating a single feedback example against an eval."""

    source: str = Field(description="Source .mb file name")
    label: Literal["PASS", "FAIL"] = Field(description="Human label (PASS=positive, FAIL=negative)")
    eval_result: Literal["PASS", "FAIL"] = Field(description="Eval verdict for this example")
    match: bool = Field(description="Whether label and eval_result agree")


class CalibrationResult(BaseModel):
    """Calibration metrics for a single eval."""

    eval_id: str = Field(description="Reference to the eval that was calibrated")
    num_examples: int = Field(description="Total number of examples evaluated")
    accuracy: float = Field(description="(TP + TN) / total")
    precision: float = Field(description="TP / (TP + FP)")
    recall: float = Field(description="TP / (TP + FN)")
    f1: float = Field(description="Harmonic mean of precision and recall")
    false_positives: int = Field(description="Eval FAIL but label PASS")
    false_negatives: int = Field(description="Eval PASS but label FAIL")
    verdict: Literal["GOOD", "WEAK", "BAD"] = Field(description="Calibration verdict")
    examples: list[CalibrationExample] = Field(default_factory=list, description="Per-example details")


class CalibrationReport(BaseModel):
    """Complete calibration report."""

    version: str = Field(default="1.0", description="File format version")
    prompt_ref: str = Field(description="Reference to the prompt file")
    eval_file: str = Field(description="Path to the eval file used")
    calibrations: list[CalibrationResult] = Field(
        default_factory=list, description="Per-eval calibration results"
    )

    def to_yaml_dict(self) -> dict:
        """Convert to dictionary suitable for YAML serialization."""
        return {
            "version": self.version,
            "prompt_ref": self.prompt_ref,
            "eval_file": self.eval_file,
            "calibrations": [
                {
                    "eval_id": c.eval_id,
                    "num_examples": c.num_examples,
                    "accuracy": c.accuracy,
                    "precision": c.precision,
                    "recall": c.recall,
                    "f1": c.f1,
                    "false_positives": c.false_positives,
                    "false_negatives": c.false_negatives,
                    "verdict": c.verdict,
                    "examples": [
                        {
                            "source": ex.source,
                            "label": ex.label,
                            "eval_result": ex.eval_result,
                            "match": ex.match,
                        }
                        for ex in c.examples
                    ],
                }
                for c in self.calibrations
            ],
        }
