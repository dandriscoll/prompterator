"""Subprocess wrapper for the naming file naming tool."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NamingError(Exception):
    """Error from naming tool."""

    pass


@dataclass
class NamingToolConfig:
    """Configuration returned by naming config."""

    prior_types: list[str]
    source_type: str
    feedback_type: str


def _find_naming_executable(configured_path: str) -> str:
    """Find the naming executable path.

    Args:
        configured_path: Path from config, or "naming" for default.

    Returns:
        Resolved path to executable.

    Raises:
        NamingError: If executable not found.
    """
    if configured_path == "naming":
        # Look for bundled naming in package
        package_dir = Path(__file__).parent.parent.parent.parent
        bundled_naming = package_dir / "bin" / "naming"
        if bundled_naming.exists():
            return str(bundled_naming)

        # Fall back to PATH
        found = shutil.which("naming")
        if found:
            return found

        raise NamingError("naming tool not found. Install prompterator or provide path in config.")

    # Custom path
    path = Path(configured_path)
    if path.exists():
        return str(path)

    raise NamingError(f"naming tool not found at configured path: {configured_path}")


def _run_naming(executable: str, args: list[str], timeout: int = 30) -> str:
    """Run naming command and return stdout.

    Args:
        executable: Path to naming executable.
        args: Command arguments.
        timeout: Timeout in seconds.

    Returns:
        stdout output.

    Raises:
        NamingError: On command failure.
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
        raise NamingError(f"naming command failed: {e.stderr or e.stdout}")
    except subprocess.TimeoutExpired:
        raise NamingError(f"naming command timed out after {timeout}s")
    except FileNotFoundError:
        raise NamingError(f"naming executable not found: {executable}")


class NamingClient:
    """Client for interacting with the naming tool."""

    def __init__(self, executable: str = "naming", timeout: int = 30):
        """Initialize naming client.

        Args:
            executable: Path to naming executable or "naming" for default.
            timeout: Command timeout in seconds.
        """
        self._executable = _find_naming_executable(executable)
        self._timeout = timeout
        self._config: NamingToolConfig | None = None
        self._descriptor: dict[str, Any] | None = None

    def descriptor(self) -> dict[str, Any]:
        """Get the Boutiques descriptor for the naming tool.

        Returns:
            Dictionary containing the Boutiques descriptor JSON.

        Raises:
            NamingError: If the tool doesn't support --descriptor.
        """
        if self._descriptor is not None:
            return self._descriptor

        output = _run_naming(self._executable, ["--descriptor"], self._timeout)
        try:
            self._descriptor = json.loads(output)
        except json.JSONDecodeError as e:
            raise NamingError(f"Invalid JSON from --descriptor: {e}")

        return self._descriptor

    def config(self) -> NamingToolConfig:
        """Get naming configuration.

        Returns:
            NamingToolConfig with type information.
        """
        if self._config is not None:
            return self._config

        output = _run_naming(self._executable, ["config"], self._timeout)

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

        self._config = NamingToolConfig(
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
        return _run_naming(self._executable, ["propose", str(path), target_type], self._timeout)

    def ready(self, path: str | Path) -> bool:
        """Check if a file is ready for transformation.

        Args:
            path: File path to check.

        Returns:
            True if file is ready, False otherwise.
        """
        output = _run_naming(self._executable, ["ready", str(path)], self._timeout)
        return output.lower() == "true"

    def bundles(self, directory: str | Path = ".") -> list[tuple[str, list[str]]]:
        """List bundles in a directory.

        Args:
            directory: Directory to list bundles from.

        Returns:
            List of (prior, [sources]) tuples.
        """
        output = _run_naming(self._executable, ["bundles", str(directory)], self._timeout)

        bundles = []
        for line in output.split("\n"):
            if not line.strip():
                continue

            parts = line.split("\t")
            prior = parts[0]
            sources = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            bundles.append((prior, sources))

        return bundles
