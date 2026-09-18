"""Built-in tool: ``web_fetch`` — fetch a URL and convert HTML to markdown."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool
from markdownify import markdownify

_DEFAULT_TIMEOUT_S = 20.0
_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB hard cap to keep tool output bounded


@tool
def web_fetch(url: str, timeout: float | None = None) -> str:
    """Fetch a URL and return its content as markdown.

    Args:
        url: The URL to fetch. Must use ``http`` or ``https``.
        timeout: Optional per-request timeout in seconds (default 20).

    Returns:
        Markdown-formatted text on success. HTML responses are converted
        via ``markdownify``; non-HTML responses are returned as plain
        text. **Error conditions return a plain-text string starting
        with ``"Error: "``** rather than raising — this keeps the
        agent's tool loop alive so the model can read the error,
        consider it, and try a different URL/tool. Cases handled this
        way: HTTP 4xx/5xx, DNS / connection failures, timeouts.

    Raises:
        ValueError: Only for caller-side mistakes the model can fix
            without a network round-trip — non-http(s) scheme, or a
            response that exceeds the 5 MiB cap.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"web_fetch requires http(s) URL, got {url!r}")

    request_timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT_S

    try:
        with httpx.Client(timeout=request_timeout, follow_redirects=True) as client:
            response = client.get(url)
            if response.is_error:
                return f"Error: HTTP {response.status_code} {response.reason_phrase} for {response.request.url}"
            content_type = response.headers.get("content-type", "").lower()
            body = response.content
            if len(body) > _MAX_BYTES:
                raise ValueError(
                    f"web_fetch response is {len(body)} bytes, exceeding the {_MAX_BYTES}-byte cap.",
                )
            text = response.text
    except httpx.HTTPError as exc:
        # Network-layer issues: DNS lookup, connection refused, timeout,
        # TLS error, …. Surface them as plain-text so the agent can
        # decide whether to retry, switch URLs, or give up gracefully
        # rather than crashing the whole graph.
        return f"Error: {type(exc).__name__}: {exc}"

    if "html" in content_type:
        return markdownify(text, heading_style="ATX").strip()
    return text.strip()
