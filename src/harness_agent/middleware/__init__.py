"""Custom middleware shipped with harness-agent."""

from __future__ import annotations

from harness_agent.middleware.client_tool_search import ClientToolSearchMiddleware
from harness_agent.middleware.filesystem_guard import FilesystemGuardMiddleware
from harness_agent.middleware.media_offload import MediaOffloadMiddleware
from harness_agent.middleware.memory import MemoryMiddleware
from harness_agent.middleware.model_settings import (
    CONFIGURABLE_MAX_INPUT_TOKENS,
    CONFIGURABLE_MODEL_SETTINGS,
    ModelSettingsMiddleware,
)
from harness_agent.middleware.native_tool_search import NativeToolSearchMiddleware
from harness_agent.middleware.peer import PeerAgentMiddleware
from harness_agent.middleware.pii import detect_pii
from harness_agent.middleware.tool_search import ToolSearchMiddleware

__all__ = [
    "CONFIGURABLE_MAX_INPUT_TOKENS",
    "CONFIGURABLE_MODEL_SETTINGS",
    "ClientToolSearchMiddleware",
    "FilesystemGuardMiddleware",
    "MediaOffloadMiddleware",
    "MemoryMiddleware",
    "ModelSettingsMiddleware",
    "NativeToolSearchMiddleware",
    "PeerAgentMiddleware",
    "ToolSearchMiddleware",
    "detect_pii",
]
