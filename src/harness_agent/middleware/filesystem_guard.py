"""Enforce filesystem path rules when deepagents permissions are unavailable."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Literal

from deepagents.backends.utils import validate_path
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from wcmatch import glob as wcglob

logger = logging.getLogger(__name__)

# Aligned with deepagents.middleware.filesystem._DEFAULT_FS_TOOL_OPS /
# _check_fs_permission / _FS_WCMATCH_FLAGS (private there; mirrored here).
_FS_TOOL_OPS: dict[str, Literal["read", "write"]] = {
    "ls": "read",
    "read_file": "read",
    "glob": "read",
    "grep": "read",
    "write_file": "write",
    "edit_file": "write",
    "delete": "write",
}
_FS_WCMATCH_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR


def _tool_base_name(name: str) -> str:
    trimmed = name.strip()
    slash = trimmed.rfind("/")
    return trimmed[slash + 1 :] if slash >= 0 else trimmed


def _path_from_tool_args(tool_name: str, params: dict[str, Any]) -> str | None:
    base = _tool_base_name(tool_name)
    if base in {"read_file", "write_file", "edit_file", "delete"}:
        for key in ("file_path", "path"):
            raw = params.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return None
    if base in {"ls", "glob", "grep"}:
        raw = params.get("path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _realpath_if_available(path: str) -> str | None:
    try:
        if os.path.lexists(path):
            return os.path.realpath(path)
    except OSError:
        return None
    return None


def _expand_path_candidates(path: str) -> tuple[str, ...]:
    """Path forms to match against deny/allow patterns.

    Includes the logical path and its realpath (macOS ``/etc`` → ``/private/etc``).
    """
    out: list[str] = [path]
    real = _realpath_if_available(path)
    if real and real not in out:
        out.append(real)
    return tuple(out)


def _expand_patterns(pattern: str) -> tuple[str, ...]:
    """Expand one policy pattern so directory roots are covered.

    ``/etc/**`` must also match ``/etc`` itself (``ls /etc``). When the prefix
    resolves through a symlink (macOS), also match the realpath forms.
    """
    patterns: list[str] = [pattern]
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        if prefix and prefix not in patterns:
            patterns.append(prefix)
        real_prefix = _realpath_if_available(prefix) if prefix else None
        if real_prefix:
            if real_prefix not in patterns:
                patterns.append(real_prefix)
            real_glob = f"{real_prefix}/**"
            if real_glob not in patterns:
                patterns.append(real_glob)
    return tuple(patterns)


def _path_matches_patterns(path: str, patterns: Iterable[str]) -> bool:
    for candidate in _expand_path_candidates(path):
        for pattern in patterns:
            for expanded in _expand_patterns(pattern):
                if wcglob.globmatch(candidate, expanded, flags=_FS_WCMATCH_FLAGS):
                    return True
    return False


def _check_fs_permission(
    rules: list[FilesystemPermission],
    operation: Literal["read", "write"],
    path: str,
) -> Literal["allow", "deny", "interrupt"]:
    """First-match path rule — directory roots and symlink aliases included."""
    for rule in rules:
        if operation not in rule.operations:
            continue
        if _path_matches_patterns(path, rule.paths):
            return rule.mode
    return "allow"


def filesystem_guard_block_reason(
    permissions: list[FilesystemPermission],
    *,
    tool_name: str,
    params: dict[str, Any],
) -> str | None:
    """Return a rejection message when *tool_name* is denied by *permissions*.

    Matched sensitive paths refuse access for read/write filesystem tools
    (``ls``, ``read_file``, ``glob``, ``grep``, ``write_file``, ``edit_file``).
    """
    if not permissions:
        return None
    base = _tool_base_name(tool_name)
    operation = _FS_TOOL_OPS.get(base)
    if operation is None:
        return None
    raw_path = _path_from_tool_args(tool_name, params)
    if raw_path is None:
        return None
    try:
        validated_path = validate_path(raw_path)
    except ValueError:
        return None
    if _check_fs_permission(permissions, operation, validated_path) == "deny":
        return f"Error: permission denied for {operation} on {validated_path}"
    return None


class FilesystemGuardMiddleware(AgentMiddleware[Any, Any]):
    """Apply SecurityPolicy filesystem rules on execution-capable backends.

    deepagents ``FilesystemMiddleware`` rejects ``permissions`` when the backend
    implements ``SandboxBackendProtocol`` (``local_shell``). Harness still needs
    deny rules for read/write filesystem tools (``ls``, ``read_file``, ``glob``,
    ``grep``, ``write_file``, ``edit_file``) there, so this middleware enforces
    the same rules at the tool-call boundary instead.
    """

    def __init__(self, permissions: list[FilesystemPermission]) -> None:
        self._permissions = list(permissions)

    def _blocked_message(self, request: ToolCallRequest) -> ToolMessage | None:
        tool_call = request.tool_call
        tool_name = str(tool_call.get("name") or "")
        raw_args: Any = tool_call.get("args") or {}
        params: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        reason = filesystem_guard_block_reason(
            self._permissions,
            tool_name=tool_name,
            params=params,
        )
        if reason is None:
            return None
        logger.info("FilesystemGuard blocked %s", tool_name)
        return ToolMessage(
            content=reason,
            tool_call_id=str(tool_call.get("id") or ""),
            status="error",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        blocked = self._blocked_message(request)
        if blocked is not None:
            return blocked
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        blocked = self._blocked_message(request)
        if blocked is not None:
            return blocked
        return handler(request)


__all__ = ["FilesystemGuardMiddleware", "filesystem_guard_block_reason"]
