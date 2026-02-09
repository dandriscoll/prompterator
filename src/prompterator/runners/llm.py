"""Subprocess wrapper for LLM runner scripts."""

import json
import shutil
import subprocess
from pathlib import Path


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
        return str(path)

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

        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or "Unknown error"
            raise LLMError(f"LLM generation failed: {error_msg}")
        except subprocess.TimeoutExpired:
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
            raise LLMError(f"Failed to get descriptor: {error_msg}")
        except subprocess.TimeoutExpired:
            raise LLMError(f"Descriptor request timed out after {timeout}s")
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON from descriptor: {e}")
        except FileNotFoundError:
            raise LLMError(f"LLM runner not found: {self._executable}")
