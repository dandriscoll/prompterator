"""Configuration schema for prompterator."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DirectoriesConfig(BaseModel):
    """Directory configuration."""

    prompts: str = Field(default=".", description="Directory for prompt files")
    feedback: str = Field(default=".", description="Directory for feedback files")
    issues: str = Field(default=".prompterator/issues", description="Directory for issue files")
    evals: str = Field(default=".prompterator/evals", description="Directory for eval files")
    results: str = Field(default=".prompterator/results", description="Directory for result files")


class LLMRoleConfig(BaseModel):
    """Base LLM configuration for a role."""

    runner: str = Field(
        default="anthropic",
        description="LLM runner: 'anthropic', 'openai', or path to custom script",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens to generate")


class AuthorConfig(LLMRoleConfig):
    """Author LLM configuration - takes a prior and produces a source."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class EditorConfig(LLMRoleConfig):
    """Editor LLM configuration - turns feedback into evals and makes changes to prompts."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class CriticConfig(LLMRoleConfig):
    """Critic LLM configuration - runs evals."""

    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")


class FeedbackConfig(BaseModel):
    """Feedback processing configuration."""

    categories: list[str] = Field(
        default_factory=lambda: ["clarity", "completeness", "accuracy", "tone", "format"],
        description="Valid feedback categories",
    )
    min_occurrences: int = Field(
        default=1,
        ge=1,
        description="Minimum occurrences to create an issue",
    )


class NamingConfig(BaseModel):
    """File naming tool configuration."""

    executable: str = Field(
        default="naming",
        description="Path to naming executable or 'naming' for default",
    )
    timeout: int = Field(default=30, gt=0, description="Command timeout in seconds")


class WorkflowConfig(BaseModel):
    """Workflow mode configuration."""

    git_mode: bool = Field(
        default=False,
        description="Enable git mode for in-place editing (overwrites files instead of creating variations)",
    )


class Config(BaseModel):
    """Root configuration for prompterator."""

    version: str = Field(default="1.0", description="Config file version")
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    author: AuthorConfig = Field(default_factory=AuthorConfig)
    editor: EditorConfig = Field(default_factory=EditorConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)

    def get_dir(self, name: Literal["prompts", "feedback", "issues", "evals", "results"], base: Path) -> Path:
        """Get resolved directory path."""
        dir_path = getattr(self.directories, name)
        path = Path(dir_path)
        if not path.is_absolute():
            path = base / path
        return path

    def to_yaml_dict(self) -> dict:
        """Convert to dictionary suitable for YAML serialization."""
        return {
            "version": self.version,
            "directories": {
                "prompts": self.directories.prompts,
                "feedback": self.directories.feedback,
                "issues": self.directories.issues,
                "evals": self.directories.evals,
                "results": self.directories.results,
            },
            "author": {
                "runner": self.author.runner,
                "temperature": self.author.temperature,
                "max_tokens": self.author.max_tokens,
            },
            "editor": {
                "runner": self.editor.runner,
                "temperature": self.editor.temperature,
                "max_tokens": self.editor.max_tokens,
            },
            "critic": {
                "runner": self.critic.runner,
                "temperature": self.critic.temperature,
                "max_tokens": self.critic.max_tokens,
            },
            "feedback": {
                "categories": self.feedback.categories,
                "min_occurrences": self.feedback.min_occurrences,
            },
            "naming": {
                "executable": self.naming.executable,
                "timeout": self.naming.timeout,
            },
            "workflow": {
                "git_mode": self.workflow.git_mode,
            },
        }
