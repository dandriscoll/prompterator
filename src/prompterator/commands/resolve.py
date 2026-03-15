"""Shared resolution logic for finding prompt, issue, and eval files."""

from pathlib import Path

from prompterator.config.schema import Config
from prompterator.core.eval_spec import load_eval_file
from prompterator.core.issue import load_issue_file
from prompterator.models.eval import EvalFile
from prompterator.models.issue import IssueFile


class ResolveError(Exception):
    """Error resolving files."""


def resolve_prompt(config: Config, base_dir: Path) -> Path | None:
    """Resolve the primary prompt from config.

    Returns the path if ``directories.prompt`` is set, else None.
    """
    if config.directories.prompt is None:
        return None
    path = Path(config.directories.prompt)
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def resolve_prompt_and_evals(
    config: Config,
    base_dir: Path,
    prompt: Path | None = None,
    evals_path: Path | None = None,
) -> tuple[Path, Path, EvalFile]:
    """Resolve prompt and eval file paths.

    Accepts any combination of prompt and evals_path — derives the
    missing one from the other, or auto-detects both.

    Returns:
        Tuple of (prompt_path, evals_path, eval_file).
    """
    if evals_path is not None:
        eval_file = load_eval_file(evals_path)
        if prompt is None:
            prompt = resolve_prompt(config, base_dir)
            if prompt is None:
                prompt = config.get_dir("prompts", base_dir) / eval_file.prompt_ref
            if not prompt.exists():
                raise ResolveError(f"Prompt file not found at {prompt}")
    elif prompt is not None:
        evals_dir = config.get_dir("evals", base_dir)
        base_name = prompt.stem.split(".")[0]
        evals_path = evals_dir / f"{base_name}.eval.yaml"
        if not evals_path.exists():
            raise ResolveError(
                f"No eval file found at {evals_path}\n"
                "Run 'prompterator evals' first or specify --evals path."
            )
        eval_file = load_eval_file(evals_path)
    else:
        # Auto-detect from first .eval.yaml
        evals_dir = config.get_dir("evals", base_dir)
        eval_files = sorted(evals_dir.glob("*.eval.yaml"))
        if not eval_files:
            raise ResolveError(
                f"No .eval.yaml files found in {evals_dir}\n"
                "Run 'prompterator evals' first, or specify a prompt or --evals path."
            )
        evals_path = eval_files[0]
        eval_file = load_eval_file(evals_path)
        prompt = resolve_prompt(config, base_dir)
        if prompt is None:
            prompt = config.get_dir("prompts", base_dir) / eval_file.prompt_ref
        if not prompt.exists():
            raise ResolveError(f"Prompt file not found at {prompt}")

    return prompt, evals_path, eval_file


def resolve_issues(
    config: Config,
    base_dir: Path,
    prompt: Path,
    issues_path: Path | None = None,
) -> tuple[Path, IssueFile]:
    """Resolve issue file path from prompt or explicit path.

    Returns:
        Tuple of (issues_path, issue_file).
    """
    if issues_path is None:
        issues_dir = config.get_dir("issues", base_dir)
        base_name = prompt.stem.split(".")[0]
        issues_path = issues_dir / f"{base_name}.issue.yaml"
        if not issues_path.exists():
            raise ResolveError(
                f"No issue file found at {issues_path}\n"
                "Run 'prompterator issues' first or specify --issues path."
            )

    issue_file = load_issue_file(issues_path)
    return issues_path, issue_file


def resolve_content(
    config: Config,
    base_dir: Path,
    cli_content: Path | None = None,
) -> list[str]:
    """Resolve content texts from CLI flag or config.

    Returns list of content strings. Empty list means no content files.
    """
    return [text for _, text in resolve_content_with_paths(config, base_dir, cli_content)]


def resolve_content_with_paths(
    config: Config,
    base_dir: Path,
    cli_content: Path | None = None,
) -> list[tuple[Path, str]]:
    """Resolve content files with their paths and texts.

    Returns list of (path, text) tuples. Empty list means no content files.
    """
    if cli_content is not None:
        return [(cli_content, cli_content.read_text())]

    raw = config.directories.content
    if raw is None:
        return []

    if isinstance(raw, str):
        raw = [raw]

    results = []
    for entry in raw:
        p = Path(entry)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            results.append((p, p.read_text()))
    return results


def resolve_counterpart(
    config: Config,
    base_dir: Path,
    cli_counterpart: Path | None = None,
) -> list[str]:
    """Resolve counterpart directions texts from CLI flag or config.

    Returns list of directions strings. Empty list means no counterpart.
    """
    return [text for _, text in resolve_counterpart_with_paths(config, base_dir, cli_counterpart)]


def resolve_counterpart_with_paths(
    config: Config,
    base_dir: Path,
    cli_counterpart: Path | None = None,
) -> list[tuple[Path, str]]:
    """Resolve counterpart directions files with their paths and texts.

    Returns list of (path, text) tuples. Empty list means no counterpart.
    """
    if cli_counterpart is not None:
        return [(cli_counterpart, cli_counterpart.read_text())]

    raw = config.directories.counterpart
    if raw is None:
        return []

    if isinstance(raw, str):
        raw = [raw]

    results = []
    for entry in raw:
        p = Path(entry)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            results.append((p, p.read_text()))
    return results


def resolve_feedback(
    config: Config,
    base_dir: Path,
    prompt_ref: str,
    feedback_dir: Path | None = None,
) -> list:
    """Resolve and parse feedback files for a prompt.

    Returns list of Feedback objects matching the given prompt_ref.
    """
    from prompterator.commands.feedback import find_mb_files, parse_mb_file

    if feedback_dir is None:
        feedback_dir = config.get_dir("feedback", base_dir)

    if not feedback_dir.exists():
        return []

    mb_files = find_mb_files(feedback_dir)
    if not mb_files:
        return []

    feedback_list = []
    for path in mb_files:
        try:
            fb = parse_mb_file(path)
            if fb.prompt_ref is None or Path(fb.prompt_ref).name == Path(prompt_ref).name:
                feedback_list.append(fb)
        except Exception:
            pass

    return feedback_list
