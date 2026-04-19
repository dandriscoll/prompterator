"""Subprocess wrapper for LLM runner scripts."""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Module-level debug state. When enabled, calls are buffered until a run
# directory is set, then flushed as individual files.
_debug_enabled: bool = False
_debug_dir: Path | None = None
_debug_buffer: list[tuple[str, str | None, str, str]] = []
_debug_context: str = ""
_debug_seq: int = 0

# Verbose mode prints full subprocess stderr/stdout and call context on failure.
_verbose_enabled: bool = False


def enable_debug_log() -> None:
    """Enable debug logging. Writes to cwd by default; overridden by set_debug_log_dir()."""
    global _debug_enabled, _debug_dir
    _debug_enabled = True
    _debug_dir = Path.cwd()


def enable_verbose() -> None:
    """Emit full subprocess stderr/stdout to stderr on LLM runner failure."""
    global _verbose_enabled
    _verbose_enabled = True


def set_debug_log_dir(directory: Path) -> None:
    """Set the debug output directory and flush any buffered calls."""
    global _debug_dir
    directory.mkdir(parents=True, exist_ok=True)
    _debug_dir = directory
    for context, system, prompt, response in _debug_buffer:
        _write_entry(context, system, prompt, response)
    _debug_buffer.clear()


def debug_context(label: str) -> None:
    """Set the current debug context label (e.g. 'improve', 'tune.3.eval')."""
    global _debug_context
    _debug_context = label


def _write_entry(context: str, system: str | None, prompt: str, response: str) -> None:
    global _debug_seq
    _debug_seq += 1
    ctx = context.replace(".", "-") if context else "unknown"
    filename = f"debug-{_debug_seq:03d}-{ctx}.log"
    path = _debug_dir / filename
    with open(path, "w") as f:
        if system:
            f.write(f"--- SYSTEM ---\n{system}\n\n")
        f.write(f"--- INPUT ---\n{prompt}\n\n")
        f.write(f"--- OUTPUT ---\n{response}\n")


def _log_call(system: str | None, prompt: str, response: str) -> None:
    """Write one LLM call to its own debug file, or buffer it."""
    if not _debug_enabled:
        return
    context = _debug_context
    if _debug_dir is not None:
        _write_entry(context, system, prompt, response)
    else:
        _debug_buffer.append((context, system, prompt, response))


def _decode(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _emit_failure(
    reason: str,
    args: list[str],
    returncode: int | None,
    stdout: str | None,
    stderr: str | None,
    prompt: str,
) -> None:
    """Print a detailed failure report to stderr for --verbose."""
    ctx = _debug_context or "(none)"
    lines = [
        "",
        f"[verbose] {reason}",
        f"  context: {ctx}",
        f"  command: {' '.join(args)}",
    ]
    if returncode is not None:
        lines.append(f"  exit:    {returncode}")
    prompt_preview = prompt if len(prompt) <= 500 else prompt[:500] + f"... [+{len(prompt) - 500} chars]"
    lines.append(f"  prompt:  {prompt_preview!r}")
    lines.append(f"  stderr:  {(stderr or '').strip() or '(empty)'}")
    lines.append(f"  stdout:  {(stdout or '').strip() or '(empty)'}")
    sys.stderr.write("\n".join(lines) + "\n")


class LLMError(Exception):
    """Error from LLM runner."""

    pass


def _find_llm_executable(runner: str) -> str:
    """Find the LLM runner executable path.

    Args:
        runner: "anthropic", "openai", "azure-openai", or path to custom script.

    Returns:
        Resolved path to executable.

    Raises:
        LLMError: If executable not found.
    """
    # Map shorthand names to bundled scripts
    if runner in ("anthropic", "openai", "azure-openai"):
        script_name = f"llm-{runner}"

        # Look for bundled script in package
        package_dir = Path(__file__).parent.parent.parent.parent
        bundled_script = package_dir / "bin" / script_name
        if bundled_script.exists():
            return str(bundled_script)

        # Fall back to PATH
        found = shutil.which(script_name)
        if found:
            return found

        raise LLMError(f"LLM runner '{script_name}' not found. Install prompterator[{runner}].")

    # Custom path
    path = Path(runner)
    if path.exists():
        return str(path.resolve())

    # Try PATH
    found = shutil.which(runner)
    if found:
        return found

    raise LLMError(f"LLM runner not found: {runner}")


class LLMClient:
    """Client for interacting with LLM runner scripts."""

    def __init__(
        self,
        runner: str = "anthropic",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            runner: "anthropic", "openai", "azure-openai", or path to custom script.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            model: Model name or deployment ID.
            endpoint: API endpoint URL.
            api_version: API version (for Azure OpenAI).
        """
        self._executable = _find_llm_executable(runner)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model = model
        self._endpoint = endpoint
        self._api_version = api_version

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 300,
        output_path: str | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            timeout: Timeout in seconds.

        Returns:
            Generated response text.

        Raises:
            LLMError: On generation failure.
        """
        args = [self._executable]

        if system:
            args.extend(["--system", system])

        temp = temperature if temperature is not None else self._temperature
        args.extend(["--temperature", str(temp)])

        tokens = max_tokens if max_tokens is not None else self._max_tokens
        args.extend(["--max-tokens", str(tokens)])

        if self._model:
            args.extend(["--model", self._model])

        if self._endpoint:
            args.extend(["--endpoint", self._endpoint])

        if self._api_version:
            args.extend(["--api-version", self._api_version])

        # Allow runners that produce sidecar artifacts (images, audio, etc.)
        # to co-locate them with the text output by exporting the intended
        # output path. Text-only runners ignore it.
        run_env = None
        if output_path is not None:
            import os as _os
            run_env = {**_os.environ, "PROMPTERATOR_OUTPUT_PATH": str(output_path)}

        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                env=run_env,
            )
            response = result.stdout.strip()
            _log_call(system, prompt, response)
            return response
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or "Unknown error"
            if _verbose_enabled:
                _emit_failure("LLM runner exited non-zero", args, e.returncode, e.stdout, e.stderr, prompt)
            raise LLMError(f"LLM generation failed: {error_msg}")
        except subprocess.TimeoutExpired as e:
            if _verbose_enabled:
                _emit_failure(
                    f"LLM runner timed out after {timeout}s",
                    args,
                    None,
                    _decode(e.stdout),
                    _decode(e.stderr),
                    prompt,
                )
            raise LLMError(f"LLM generation timed out after {timeout}s")
        except FileNotFoundError:
            raise LLMError(f"LLM runner not found: {self._executable}")

    def descriptor(self, timeout: int = 30) -> dict:
        """Get the Boutiques descriptor from the runner.

        Returns:
            Parsed JSON descriptor dict.

        Raises:
            LLMError: If the runner fails or returns invalid JSON.
        """
        try:
            result = subprocess.run(
                [self._executable, "--descriptor"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or "Unknown error"
            if _verbose_enabled:
                _emit_failure(
                    "descriptor request failed",
                    [self._executable, "--descriptor"],
                    e.returncode,
                    e.stdout,
                    e.stderr,
                    "",
                )
            raise LLMError(f"Failed to get descriptor: {error_msg}")
        except subprocess.TimeoutExpired as e:
            if _verbose_enabled:
                _emit_failure(
                    f"descriptor request timed out after {timeout}s",
                    [self._executable, "--descriptor"],
                    None,
                    _decode(e.stdout),
                    _decode(e.stderr),
                    "",
                )
            raise LLMError(f"Descriptor request timed out after {timeout}s")
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON from descriptor: {e}")
        except FileNotFoundError:
            raise LLMError(f"LLM runner not found: {self._executable}")
