"""Feedback command - parse and display .mb feedback files."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.models.feedback import Feedback, FeedbackEntry


def _extract_prior_prompt_ref(raw_content: str) -> str | None:
    """Extract prompt file reference from @prior directives in raw .mb content.

    The markback library only stores one @prior per record (the last one),
    but .mb files can have multiple @prior lines per block.  We need to
    scan the raw text to find the @prior that points to the prompt file
    (.prompt.txt or .prompt.md) rather than the input data file.
    """
    for line in raw_content.splitlines():
        line = line.strip()
        if line.startswith("@prior "):
            value = line.split(" ", 1)[1].strip()
            if value.endswith(".prompt.txt") or value.endswith(".prompt.md"):
                return value
    return None


def parse_mb_file(path: Path) -> Feedback:
    """Parse a markback file into Feedback model.

    Uses the markback library to parse .mb files.
    Markback format: <<< feedback_text
    """
    try:
        import markback
    except ImportError:
        click.echo("Error: markback package not installed", err=True)
        click.echo("Install with: pip install markback", err=True)
        raise SystemExit(1)

    with open(path) as f:
        content = f.read()

    # Parse with markback
    result = markback.parse_string(content, source_file=path)

    entries = []

    # The markback library only stores one @prior per record (the last).
    # Scan raw content to find the @prior that references the prompt file.
    prompt_ref = _extract_prior_prompt_ref(content)

    # Extract feedback from markback records
    for record in result.records:
        feedback_text = record.feedback
        if not feedback_text:
            continue

        # Fall back to @source if no prompt prior was found in raw content
        if not prompt_ref and hasattr(record, "source") and record.source:
            source_val = getattr(record.source, "value", None) or str(record.source)
            if source_val:
                prompt_ref = source_val

        # Check for ref= in the feedback
        if "ref=" in feedback_text:
            # Extract ref and remove from feedback
            parts = feedback_text.split(";")
            for i, part in enumerate(parts):
                if "ref=" in part:
                    ref_part = part.strip()
                    prompt_ref = ref_part.split("=", 1)[1].strip()
                    parts.pop(i)
                    break
            feedback_text = ";".join(parts)

        feedback_text = feedback_text.strip()
        if feedback_text:
            entries.append(FeedbackEntry(text=feedback_text))

    # Try to infer prompt_ref from filename if not found
    if not prompt_ref:
        stem = path.stem
        # Try common patterns
        for ext in [".prompt.txt", ".prompt.md", ".out.txt"]:
            potential_ref = stem + ext
            if (path.parent / potential_ref).exists():
                prompt_ref = potential_ref
                break

    return Feedback(
        source_file=str(path),
        prompt_ref=prompt_ref,
        entries=entries,
        raw_content=content,
    )


def find_mb_files(directory: Path) -> list[Path]:
    """Find all .mb files in a directory."""
    return sorted(directory.glob("*.mb"))


@click.command("feedback")
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for .mb files (default: from config)",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def feedback_cmd(directory: Path | None, as_json: bool) -> None:
    """Parse and display .mb feedback files."""
    config = load_config()
    base_dir = get_config_base_dir()

    if directory is None:
        directory = config.get_dir("feedback", base_dir)

    mb_files = find_mb_files(directory)

    if not mb_files:
        click.echo(f"No .mb files found in {directory}")
        return

    all_feedback = []
    for path in mb_files:
        try:
            feedback = parse_mb_file(path)
            all_feedback.append(feedback)
        except Exception as e:
            click.echo(f"Error parsing {path}: {e}", err=True)

    if as_json:
        import json

        output = [
            {
                "source_file": f.source_file,
                "prompt_ref": f.prompt_ref,
                "entries": [{"text": e.text} for e in f.entries],
            }
            for f in all_feedback
        ]
        click.echo(json.dumps(output, indent=2))
    else:
        for feedback in all_feedback:
            click.echo(f"\n{feedback.source_file}")
            if feedback.prompt_ref:
                click.echo(f"  Prompt: {feedback.prompt_ref}")
            click.echo(f"  Entries: {len(feedback.entries)}")
            for entry in feedback.entries:
                click.echo(f"    - {entry.text}")

        click.echo(f"\nTotal: {len(all_feedback)} feedback files")
