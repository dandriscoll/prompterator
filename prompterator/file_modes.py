from __future__ import annotations

from pathlib import Path


def resolve_file_mode(file_mode: str, cwd: Path) -> str:
    normalized = file_mode.strip().lower()
    if normalized == "auto":
        return "git" if is_git_repo(cwd) else "plain"
    if normalized in {"git", "plain"}:
        return normalized
    raise ValueError(f"Unsupported FILE_MODE: {file_mode}")


def is_git_repo(start: Path) -> bool:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return True
    return False


def write_output(path: Path, content: str, file_mode: str, suffix: str) -> Path:
    target = Path(path)
    if file_mode == "git":
        destination = target
    else:
        destination = ensure_unique(derive_revision_path(target, suffix))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def derive_revision_path(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Unable to find available filename for {path}")
