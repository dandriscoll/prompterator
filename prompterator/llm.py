from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    pass


def invoke_llm(endpoint: str, prompt: str, api_key: Optional[str], timeout: int) -> str:
    payload = json.dumps(_build_payload(endpoint, prompt)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        if _is_azure_endpoint(endpoint):
            headers["api-key"] = api_key
        else:
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


def _build_payload(endpoint: str, prompt: str) -> Dict[str, Any]:
    if _is_chat_endpoint(endpoint):
        return {"messages": [{"role": "user", "content": prompt}]}
    return {"prompt": prompt}


def _is_chat_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return "/chat/completions" in parsed.path.lower()


def _is_azure_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    host = parsed.netloc.lower()
    return host.endswith(".openai.azure.com") or "/openai/deployments/" in parsed.path.lower()


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
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    raise LLMError("LLM response missing expected text field")
