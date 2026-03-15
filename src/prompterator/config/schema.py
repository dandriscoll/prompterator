"""Configuration schema for prompterator."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DirectoriesConfig(BaseModel):
    """Directory configuration."""

    prompts: str = Field(default=".", description="Directory for prompt files")
    feedback: str = Field(default=".", description="Directory for feedback files")
    issues: str = Field(default=".", description="Directory for issue files")
    evals: str = Field(default=".", description="Directory for eval files")
    results: str = Field(default=".", description="Directory for result files")
    prompt: str | None = Field(default=None, description="Path to the primary prompt file")
    content: str | list[str] | None = Field(
        default=None,
        description="Content file(s) to pair with the prompt. Can be a single path or a list.",
    )


class StackConfig(BaseModel):
    """Named LLM connection stack."""

    runner: str = Field(
        default="anthropic",
        description="LLM runner: 'anthropic', 'openai', 'azure-openai', or path to custom script",
    )
    model: str | None = Field(
        default=None,
        description="Model name or deployment ID (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')",
    )
    endpoint: str | None = Field(
        default=None,
        description="API endpoint URL (overrides environment variable)",
    )
    api_version: str | None = Field(
        default=None,
        description="API version (for Azure OpenAI)",
    )


class LLMRoleConfig(BaseModel):
    """Base LLM configuration for a role."""

    stack: str = Field(
        default="default",
        description="Name of the stack to use for LLM connection settings",
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
    """Critic configuration - runs evals.

    Supports two modes:
    - "llm" (default): Uses an LLM to evaluate prompts via rubric/assertion.
    - "script": Runs an external script that receives eval input as YAML on stdin
      and produces eval results as YAML on stdout.

    Script mode input (YAML on stdin):
        prompt: <prompt text>
        eval:
            id: <eval id>
            type: rubric | assertion
            rubric:                    # present if type=rubric
                criteria: [...]
                scoring: all_required | any_required | weighted
                weights: [...]         # optional
            assertion: <text>          # present if type=assertion
            description: <text>        # optional

    Script mode output (YAML on stdout):
        eval_id: <eval id>
        passed: true | false
        score: <0.0-1.0>
        details: <optional string>
    """

    mode: Literal["llm", "script"] = Field(
        default="llm",
        description="Critic mode: 'llm' for LLM-based evaluation, 'script' for external script",
    )
    script: str | None = Field(
        default=None,
        description="Path to critic script executable (required when mode='script')",
    )
    script_timeout: int = Field(
        default=60,
        gt=0,
        description="Timeout in seconds for script execution",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    samples: int = Field(
        default=3,
        ge=1,
        description="Number of eval samples per test run",
    )
    ensemble: int = Field(
        default=5,
        ge=1,
        description="Number of ensemble critic evaluations per output per eval",
    )
    confidence_threshold: float = Field(
        default=9.0,
        ge=0.0,
        le=10.0,
        description="Score out of 10 required for an eval to be considered passing",
    )

    @model_validator(mode="after")
    def _validate_script_mode(self) -> "CriticConfig":
        if self.mode == "script" and not self.script:
            raise ValueError("critic.script is required when mode='script'")
        return self


class FeedbackConfig(BaseModel):
    """Feedback processing configuration."""

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
    stacks: dict[str, StackConfig] = Field(
        default_factory=lambda: {"default": StackConfig()},
        description="Named LLM connection stacks",
    )
    author: AuthorConfig = Field(default_factory=AuthorConfig)
    editor: EditorConfig = Field(default_factory=EditorConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)

    @model_validator(mode="after")
    def _validate_stack_references(self) -> "Config":
        for role_name in ("author", "editor", "critic"):
            role = getattr(self, role_name)
            if role.stack not in self.stacks:
                raise ValueError(
                    f"{role_name}.stack references unknown stack '{role.stack}'; "
                    f"available stacks: {', '.join(sorted(self.stacks))}"
                )
        return self

    def resolve_role(self, name: str) -> dict:
        """Merge stack fields + role fields into a flat dict.

        Returns dict with keys: runner, model, endpoint, api_version,
        temperature, max_tokens.
        """
        role: LLMRoleConfig = getattr(self, name)
        stack = self.stacks[role.stack]
        return {
            "runner": stack.runner,
            "model": stack.model,
            "endpoint": stack.endpoint,
            "api_version": stack.api_version,
            "temperature": role.temperature,
            "max_tokens": role.max_tokens,
        }

    def get_dir(self, name: Literal["prompts", "feedback", "issues", "evals", "results"], base: Path) -> Path:
        """Get resolved directory path."""
        dir_path = getattr(self.directories, name)
        path = Path(dir_path)
        if not path.is_absolute():
            path = base / path
        return path

    @staticmethod
    def _role_to_dict(role: LLMRoleConfig) -> dict:
        """Convert role config to dict for YAML serialization."""
        d: dict = {
            "stack": role.stack,
            "temperature": role.temperature,
            "max_tokens": role.max_tokens,
        }
        return d

    @staticmethod
    def _stack_to_dict(stack: StackConfig) -> dict:
        """Convert stack config to dict, omitting None values."""
        d: dict = {"runner": stack.runner}
        if stack.model is not None:
            d["model"] = stack.model
        if stack.endpoint is not None:
            d["endpoint"] = stack.endpoint
        if stack.api_version is not None:
            d["api_version"] = stack.api_version
        return d

    def to_yaml_dict(self) -> dict:
        """Convert to dictionary suitable for YAML serialization."""
        return {
            "version": self.version,
            "directories": {
                **({"prompt": self.directories.prompt} if self.directories.prompt else {}),
                "prompts": self.directories.prompts,
                "feedback": self.directories.feedback,
                "issues": self.directories.issues,
                "evals": self.directories.evals,
                "results": self.directories.results,
            },
            "stacks": {
                name: self._stack_to_dict(stack)
                for name, stack in self.stacks.items()
            },
            "author": self._role_to_dict(self.author),
            "editor": self._role_to_dict(self.editor),
            "critic": {
                **self._role_to_dict(self.critic),
                "mode": self.critic.mode,
                **({"script": self.critic.script} if self.critic.script else {}),
                **({"script_timeout": self.critic.script_timeout} if self.critic.script_timeout != 60 else {}),
                "samples": self.critic.samples,
                "confidence_threshold": self.critic.confidence_threshold,
            },
            "feedback": {
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
