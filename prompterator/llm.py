from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    pass


def invoke_llm(endpoint: str, prompt: str, api_key: Optional[str], timeout: int) -> str:
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=payload, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise LLMError(f"LLM request failed ({exc.code}): {error_body}") from exc
    except URLError as exc:
        raise LLMError(f"LLM request failed: {exc.reason}") from exc

    return _extract_output(body)


def _extract_output(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()

    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return _extract_from_dict(payload)
    raise LLMError("Unexpected LLM response format")


def _extract_from_dict(payload: Dict[str, Any]) -> str:
    for key in ("text", "output", "response", "result"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    raise LLMError("LLM response missing expected text field")
