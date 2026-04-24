"""Issue models for consolidated feedback."""

from typing import Literal

from pydantic import BaseModel, Field


class IssueEvidence(BaseModel):
    """Evidence supporting an issue from feedback sources.

    `feedback` is the raw observation text; `instance` is the concrete
    manifestation distilled from it (an anchor), preserving author intent
    when the clustering step generalizes.
    """

    source: str = Field(description="Source file path")
    feedback: str = Field(description="Relevant feedback text")
    instance: str | None = Field(
        default=None,
        description="Concrete manifestation of the issue extracted from the feedback",
    )
    confidence: Literal["high", "medium", "low"] | None = Field(
        default=None,
        description="LLM confidence that this evidence supports the issue",
    )


class Issue(BaseModel):
    """A single consolidated issue from feedback."""

    id: str = Field(description="Unique issue identifier")
    category: str = Field(description="Issue category")
    severity: str = Field(description="Issue severity (high, medium, low)")
    summary: str = Field(description="Brief summary of the issue")
    evidence: list[IssueEvidence] = Field(default_factory=list, description="Supporting evidence")


class IssueFile(BaseModel):
    """Complete issue file structure."""

    version: str = Field(default="1.0", description="File format version")
    prompt_ref: str = Field(description="Reference to the prompt file")
    issues: list[Issue] = Field(default_factory=list, description="List of issues")

    def to_yaml_dict(self) -> dict:
        """Convert to dictionary suitable for YAML serialization."""
        return {
            "version": self.version,
            "prompt_ref": self.prompt_ref,
            "issues": [
                {
                    "id": issue.id,
                    "category": issue.category,
                    "severity": issue.severity,
                    "summary": issue.summary,
                    "evidence": [
                        {
                            "source": e.source,
                            "feedback": e.feedback,
                            **({"instance": e.instance} if e.instance else {}),
                            **({"confidence": e.confidence} if e.confidence else {}),
                        }
                        for e in issue.evidence
                    ],
                }
                for issue in self.issues
            ],
        }
