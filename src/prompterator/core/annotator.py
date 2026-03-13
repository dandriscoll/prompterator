"""Annotator - generate markback (.mb) feedback files from structured input."""

from pathlib import Path


def build_mb_block(source: str, priors: list[str], feedback: str) -> str:
    """Build a single markback block.

    Args:
        source: Value for the @source directive.
        priors: Values for @prior directives (one or more).
        feedback: The observation text.

    Returns:
        A formatted markback block string.
    """
    lines = [f"@source {source}"]
    for prior in priors:
        lines.append(f"@prior {prior}")
    lines.append(f"<<< {feedback}")
    return "\n".join(lines)


def build_mb_content(source: str, priors: list[str], observations: list[str]) -> str:
    """Build complete markback content from a source, priors, and observations.

    Args:
        source: Value for the @source directive.
        priors: Values for @prior directives.
        observations: List of feedback observation strings.

    Returns:
        Complete markback file content with blocks separated by ---.
    """
    blocks = [build_mb_block(source, priors, obs) for obs in observations]
    return "\n\n---\n\n".join(blocks) + "\n"


def derive_mb_path(source: str, output_dir: Path | None = None) -> Path:
    """Derive a .mb file path from a source path.

    Strips directory prefix and replaces all extensions with .mb.
    Example: outputs/001-r1.out.md -> 001-r1.mb

    Args:
        source: The source file path string.
        output_dir: Directory for the .mb file (default: current directory).

    Returns:
        Path for the .mb file.
    """
    name = Path(source).name
    # Strip all extensions after the stem base (e.g., 001-r1.out.md -> 001-r1)
    stem = name.split(".")[0]
    mb_name = f"{stem}.mb"
    if output_dir:
        return output_dir / mb_name
    return Path(mb_name)


def build_editor_template(source: str, priors: list[str]) -> str:
    """Build a template for interactive editing.

    Returns a string with instructions and placeholder lines.
    Lines starting with # are stripped when parsing.

    Args:
        source: The source file path.
        priors: Prior file paths.

    Returns:
        Editor template string.
    """
    lines = [
        "# Annotate: write one observation per line.",
        "# Lines starting with # are ignored. Empty lines are ignored.",
        f"# Source: {source}",
    ]
    for prior in priors:
        lines.append(f"# Prior:  {prior}")
    lines.append("#")
    lines.append("# Examples:")
    lines.append("#   opens with a conversational paragraph that pollutes the output")
    lines.append("#   replaced markdown checkboxes with priority-grouped sections")
    lines.append("#")
    lines.append("")
    return "\n".join(lines)


def parse_editor_result(text: str) -> list[str]:
    """Parse observations from editor output.

    Strips comment lines (starting with #) and blank lines.
    Each remaining non-empty line is one observation.

    Args:
        text: Raw editor output.

    Returns:
        List of observation strings.
    """
    observations = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            observations.append(stripped)
    return observations
