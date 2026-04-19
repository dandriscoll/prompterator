"""Collect command - gather mb files with their file and input refs."""

import shutil
from pathlib import Path

import click


def parse_mb_directives(mb_path: Path) -> tuple[str | None, str | None]:
    """Parse @file and @input directives from an mb file.

    Uses markback library when possible, falls back to manual parsing
    for files without feedback lines.

    Returns:
        Tuple of (file_value, input_value) - values may be None if not found.
    """
    try:
        import markback
    except ImportError:
        click.echo("Error: markback package not installed", err=True)
        click.echo("Install with: pip install markback", err=True)
        raise SystemExit(1)

    result = markback.parse_file(mb_path)

    # If markback found records, use them
    if result.records:
        rec = result.records[0]
        file_value = rec.file.value if rec.file else None
        input_value = None
        if rec.input:
            input_value = rec.input.value
        if file_value or input_value:
            # Only return early if we got at least one value from markback
            # Otherwise fall through to manual parsing
            if not input_value:
                # Try manual parsing for input
                content = mb_path.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("@input "):
                        input_value = line[7:].strip()
                        break
            return file_value, input_value

    # Fall back to manual parsing for files without feedback lines
    content = mb_path.read_text()

    file_value = None
    input_value = None

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("@file "):
            file_value = line[6:].strip()
        elif line.startswith("@input "):
            input_value = line[7:].strip()

    return file_value, input_value


def get_file_ref(mb_path: Path) -> Path | None:
    """Extract file reference from an mb file.

    Looks for @file directive in the mb file.
    """
    file_value, _ = parse_mb_directives(mb_path)

    if file_value:
        return mb_path.parent / file_value

    return None


def get_input_ref(mb_path: Path) -> Path | None:
    """Extract input reference from an mb file.

    Looks for @input directive in the mb file.
    """
    _, input_value = parse_mb_directives(mb_path)

    if input_value:
        return mb_path.parent / input_value

    return None


def collect_files(
    mb_path: Path,
    dest_dir: Path,
    dry_run: bool = False,
) -> tuple[list[Path], list[str]]:
    """Collect an mb file and its related @file and @input targets.

    Returns:
        Tuple of (copied_files, warnings)
    """
    copied = []
    warnings = []

    # Ensure mb file exists
    if not mb_path.exists():
        warnings.append(f"mb file not found: {mb_path}")
        return copied, warnings

    # Get file and input targets
    file_path = get_file_ref(mb_path)
    input_path = get_input_ref(mb_path)

    # Collect the files
    files_to_copy = [
        ("mb", mb_path),
    ]

    if file_path:
        if file_path.exists():
            files_to_copy.append(("file", file_path))
        else:
            warnings.append(f"@file target not found: {file_path}")
    else:
        warnings.append(f"no @file directive in: {mb_path.name}")

    if input_path:
        if input_path.exists():
            files_to_copy.append(("input", input_path))
        else:
            warnings.append(f"@input target not found: {input_path}")
    else:
        warnings.append(f"no @input directive in: {mb_path.name}")

    # Copy files
    for file_type, path in files_to_copy:
        dest_path = dest_dir / path.name

        if dry_run:
            click.echo(f"  [{file_type}] {path} -> {dest_path}")
        else:
            # Avoid overwriting if source and dest are the same
            if path.resolve() != dest_path.resolve():
                shutil.copy2(path, dest_path)
            copied.append(dest_path)

    return copied, warnings


@click.command("collect")
@click.argument(
    "files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--to",
    "dest_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=".",
    help="Destination directory (default: current directory)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be copied without copying",
)
def collect_cmd(
    files: tuple[Path, ...],
    dest_dir: Path,
    dry_run: bool,
) -> None:
    """Collect mb files with their @file and @input targets.

    FILES are paths to .mb files (supports shell globs like *.mb).

    For each mb file, this command finds:
    - The file target (from @file directive in the mb file)
    - The input target (from @input directive in the mb file)

    And copies all three to the destination directory.

    Examples:
        prompterator collect feedback/*.mb
        prompterator collect *.mb --to ./collected
    """
    if not files:
        click.echo("No files specified")
        raise SystemExit(1)

    # Ensure destination directory exists
    dest_dir = Path(dest_dir)
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_warnings = []

    for mb_path in files:
        if not mb_path.suffix == ".mb":
            click.echo(f"Skipping non-mb file: {mb_path}")
            continue

        click.echo(f"\nCollecting: {mb_path}")
        copied, warnings = collect_files(mb_path, dest_dir, dry_run)
        total_copied += len(copied)
        total_warnings.extend(warnings)

        for warning in warnings:
            click.echo(f"  Warning: {warning}", err=True)

    click.echo()
    if dry_run:
        click.echo("Dry run complete (no files copied)")
    else:
        click.echo(f"Copied {total_copied} file(s) to {dest_dir}")

    if total_warnings:
        click.echo(f"Warnings: {len(total_warnings)}")
