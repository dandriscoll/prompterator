"""Subprocess wrapper for the ft file naming tool."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FTError(Exception):
    """Error from ft tool."""

    pass


@dataclass
class FTConfig:
    """Configuration returned by ft config."""

    prior_types: list[str]
    source_type: str
    feedback_type: str


def _find_ft_executable(configured_path: str) -> str:
    """Find the ft executable path.

    Args:
        configured_path: Path from config, or "ft" for default.

    Returns:
        Resolved path to executable.

    Raises:
        FTError: If executable not found.
    """
    if configured_path == "ft":
        # Look for bundled ft in package
        package_dir = Path(__file__).parent.parent.parent.parent
        bundled_ft = package_dir / "bin" / "ft"
        if bundled_ft.exists():
            return str(bundled_ft)

        # Fall back to PATH
        found = shutil.which("ft")
        if found:
            return found

        raise FTError("ft tool not found. Install prompterator or provide path in config.")

    # Custom path
    path = Path(configured_path)
    if path.exists():
        return str(path)

    raise FTError(f"ft tool not found at configured path: {configured_path}")


def _run_ft(executable: str, args: list[str], timeout: int = 30) -> str:
    """Run ft command and return stdout.

    Args:
        executable: Path to ft executable.
        args: Command arguments.
        timeout: Timeout in seconds.

    Returns:
        stdout output.

    Raises:
        FTError: On command failure.
    """
    try:
        result = subprocess.run(
            [executable] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise FTError(f"ft command failed: {e.stderr or e.stdout}")
    except subprocess.TimeoutExpired:
        raise FTError(f"ft command timed out after {timeout}s")
    except FileNotFoundError:
        raise FTError(f"ft executable not found: {executable}")


class FTClient:
    """Client for interacting with the ft tool."""

    def __init__(self, executable: str = "ft", timeout: int = 30):
        """Initialize FT client.

        Args:
            executable: Path to ft executable or "ft" for default.
            timeout: Command timeout in seconds.
        """
        self._executable = _find_ft_executable(executable)
        self._timeout = timeout
        self._config: FTConfig | None = None
        self._descriptor: dict[str, Any] | None = None

    def descriptor(self) -> dict[str, Any]:
        """Get the Boutiques descriptor for the ft tool.

        Returns:
            Dictionary containing the Boutiques descriptor JSON.

        Raises:
            FTError: If the tool doesn't support --descriptor.
        """
        if self._descriptor is not None:
            return self._descriptor

        output = _run_ft(self._executable, ["--descriptor"], self._timeout)
        try:
            self._descriptor = json.loads(output)
        except json.JSONDecodeError as e:
            raise FTError(f"Invalid JSON from --descriptor: {e}")

        return self._descriptor

    def config(self) -> FTConfig:
        """Get ft configuration.

        Returns:
            FTConfig with type information.
        """
        if self._config is not None:
            return self._config

        output = _run_ft(self._executable, ["config"], self._timeout)

        prior_types = []
        source_type = ""
        feedback_type = ""

        for line in output.split("\n"):
            if line.startswith("prior-types:"):
                prior_types = [t.strip() for t in line.split(":", 1)[1].split(",")]
            elif line.startswith("source-type:"):
                source_type = line.split(":", 1)[1].strip()
            elif line.startswith("feedback-type:"):
                feedback_type = line.split(":", 1)[1].strip()

        self._config = FTConfig(
            prior_types=prior_types,
            source_type=source_type,
            feedback_type=feedback_type,
        )
        return self._config

    def propose(self, path: str | Path, target_type: str) -> str:
        """Propose a new filename for the given path with target type.

        Args:
            path: Source file path.
            target_type: Target type (e.g., "out.txt", "prompt.txt").

        Returns:
            Proposed filename path.
        """
        return _run_ft(self._executable, ["propose", str(path), target_type], self._timeout)

    def ready(self, path: str | Path) -> bool:
        """Check if a file is ready for transformation.

        Args:
            path: File path to check.

        Returns:
            True if file is ready, False otherwise.
        """
        output = _run_ft(self._executable, ["ready", str(path)], self._timeout)
        return output.lower() == "true"

    def bundles(self, directory: str | Path = ".") -> list[tuple[str, list[str]]]:
        """List bundles in a directory.

        Args:
            directory: Directory to list bundles from.

        Returns:
            List of (prior, [sources]) tuples.
        """
        output = _run_ft(self._executable, ["bundles", str(directory)], self._timeout)

        bundles = []
        for line in output.split("\n"):
            if not line.strip():
                continue

            parts = line.split("\t")
            prior = parts[0]
            sources = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            bundles.append((prior, sources))

        return bundles
