"""Annotator - generate markback (.mb) feedback files from structured input."""

from pathlib import Path


def build_mb_block(file_ref: str, inputs: list[str], feedback: str) -> str:
    """Build a single markback block.

    Args:
        file_ref: Value for the @file directive (the output being reviewed).
        inputs: Values for @input directives (one or more).
        feedback: The observation text.

    Returns:
        A formatted markback block string.
    """
    lines = [f"@file {file_ref}"]
    for inp in inputs:
        lines.append(f"@input {inp}")
    lines.append(f"<<< {feedback}")
    return "\n".join(lines)


def build_mb_content(file_ref: str, inputs: list[str], observations: list[str]) -> str:
    """Build complete markback content from a file ref, inputs, and observations.

    Args:
        file_ref: Value for the @file directive.
        inputs: Values for @input directives.
        observations: List of feedback observation strings.

    Returns:
        Complete markback file content with blocks separated by ---.
    """
    blocks = [build_mb_block(file_ref, inputs, obs) for obs in observations]
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
    stem = name.split(".")[0]
    mb_name = f"{stem}.mb"
    if output_dir:
        return output_dir / mb_name
    return Path(mb_name)


def build_editor_template(file_ref: str, inputs: list[str]) -> str:
    """Build a template for interactive editing.

    Returns a string with instructions and placeholder lines.
    Lines starting with # are stripped when parsing.

    Args:
        file_ref: The output file path.
        inputs: Input file paths.

    Returns:
        Editor template string.
    """
    lines = [
        "# Annotate: write one observation per line.",
        "# Lines starting with # are ignored. Empty lines are ignored.",
        f"# File:  {file_ref}",
    ]
    for inp in inputs:
        lines.append(f"# Input: {inp}")
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
