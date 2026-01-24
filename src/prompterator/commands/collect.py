"""Collect command - gather mb files with their source and prior files."""

import shutil
from pathlib import Path

import click


def parse_mb_directives(mb_path: Path) -> tuple[str | None, str | None]:
    """Parse @source and @prior directives from an mb file.

    Uses markback library when possible, falls back to manual parsing
    for files without feedback lines.

    Returns:
        Tuple of (source_value, prior_value) - values may be None if not found.
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
        source_value = rec.source.value if rec.source else None
        prior_value = rec.prior.value if rec.prior else None
        return source_value, prior_value

    # Fall back to manual parsing for files without feedback lines
    content = mb_path.read_text()

    source_value = None
    prior_value = None

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("@source "):
            source_value = line[8:].strip()
        elif line.startswith("@prior "):
            prior_value = line[7:].strip()

    return source_value, prior_value


def get_source_file(mb_path: Path) -> Path | None:
    """Extract source file reference from an mb file.

    Looks for @source directive in the mb file.
    """
    source_value, _ = parse_mb_directives(mb_path)

    if source_value:
        return mb_path.parent / source_value

    return None


def get_prior_file(mb_path: Path) -> Path | None:
    """Extract prior file reference from an mb file.

    Looks for @prior directive in the mb file.
    """
    _, prior_value = parse_mb_directives(mb_path)

    if prior_value:
        return mb_path.parent / prior_value

    return None


def collect_files(
    mb_path: Path,
    dest_dir: Path,
    dry_run: bool = False,
) -> tuple[list[Path], list[str]]:
    """Collect an mb file and its related source/prior files.

    Returns:
        Tuple of (copied_files, warnings)
    """
    copied = []
    warnings = []

    # Ensure mb file exists
    if not mb_path.exists():
        warnings.append(f"mb file not found: {mb_path}")
        return copied, warnings

    # Get source and prior files
    source_path = get_source_file(mb_path)
    prior_path = get_prior_file(mb_path)

    # Collect the files
    files_to_copy = [
        ("mb", mb_path),
    ]

    if source_path:
        if source_path.exists():
            files_to_copy.append(("source", source_path))
        else:
            warnings.append(f"source file not found: {source_path}")
    else:
        warnings.append(f"no @source directive in: {mb_path.name}")

    if prior_path:
        if prior_path.exists():
            files_to_copy.append(("prior", prior_path))
        else:
            warnings.append(f"prior file not found: {prior_path}")
    else:
        warnings.append(f"no @prior directive in: {mb_path.name}")

    # Copy files
    for file_type, file_path in files_to_copy:
        dest_path = dest_dir / file_path.name

        if dry_run:
            click.echo(f"  [{file_type}] {file_path} -> {dest_path}")
        else:
            # Avoid overwriting if source and dest are the same
            if file_path.resolve() != dest_path.resolve():
                shutil.copy2(file_path, dest_path)
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
    """Collect mb files with their source and prior files.

    FILES are paths to .mb files (supports shell globs like *.mb).

    For each mb file, this command finds:
    - The source file (from @source directive in the mb file)
    - The prior file (from @prior directive in the mb file)

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
