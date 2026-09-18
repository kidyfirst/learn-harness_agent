"""Peer-call helpers: one-shot requests and user-id coercion."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from harness_agent.request import ChatRequest

_PEER_THREAD_SEP = "~"


def coerce_user_id(user_raw: object) -> str | int | None:
    """Normalize ``configurable.user`` / request user to ``str | int | None``."""
    if user_raw is None:
        return None
    if isinstance(user_raw, int):
        return user_raw
    if isinstance(user_raw, str):
        return int(user_raw) if user_raw.isdigit() else user_raw
    return str(user_raw)


def derive_peer_thread_id(source_thread_id: str, to_agent_id: str) -> str:
    """Stable callee thread id: caller thread plus ``~<callee>``.

    Repeating a call to the same peer keeps the same id (the suffix is
    replaced rather than stacked).
    """
    src = source_thread_id.strip()
    dest = to_agent_id.strip()
    if not src or not dest:
        return src
    suffix = f"{_PEER_THREAD_SEP}{dest}"
    if src.endswith(suffix):
        return src
    return f"{src}{suffix}"


@dataclass(frozen=True)
class PeerCall:
    """Inputs for a host hook around ``ask_agent`` / inbox peer invocation."""

    from_agent_id: str
    to_agent_id: str
    user_id: str | int
    message: str
    source_thread_id: str | None
    source_session_key: str | None


@dataclass
class PeerSession:
    """Host-resolved callee session. ``None`` fields keep harness defaults."""

    thread_id: str | None = None
    session_key: str | None = None


def build_one_shot_request(
    *,
    user_id: str | int,
    agent_id: str,
    text: str,
    source: str,
    thread_id: str | None = None,
    session_key: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> ChatRequest:
    """Build a one-call request for *agent_id*.

    A fresh ``thread_id`` is generated when not supplied; pass an explicit
    id (typically :func:`derive_peer_thread_id`) to continue a peer thread.
    The callee sees only *text* — not the caller's conversation history.
    """
    extra = dict(configurable or {})
    if session_key:
        extra["session_key"] = session_key
    return ChatRequest(
        messages=text,
        thread_id=thread_id or uuid.uuid4().hex,
        user=str(user_id),
        agent_id=agent_id,
        source=source,
        configurable=extra or None,
    )


__all__ = [
    "PeerCall",
    "PeerSession",
    "build_one_shot_request",
    "coerce_user_id",
    "derive_peer_thread_id",
]
