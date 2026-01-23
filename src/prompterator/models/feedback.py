"""Feedback models for markback parsing."""

from pydantic import BaseModel, Field


class FeedbackEntry(BaseModel):
    """A single feedback entry from a markback file."""

    category: str = Field(description="Feedback category (e.g., clarity, completeness)")
    value: str = Field(description="The feedback value or rating")
    details: str | None = Field(default=None, description="Additional details or notes")


class Feedback(BaseModel):
    """Parsed feedback from a markback file."""

    source_file: str = Field(description="Path to the source .mb file")
    prompt_ref: str | None = Field(default=None, description="Reference to associated prompt file")
    entries: list[FeedbackEntry] = Field(default_factory=list, description="List of feedback entries")
    raw_content: str = Field(default="", description="Raw markback content")

    @property
    def categories(self) -> list[str]:
        """Get unique categories from all entries."""
        return list(set(e.category for e in self.entries))
