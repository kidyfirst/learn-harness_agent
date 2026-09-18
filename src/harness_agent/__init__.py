"""orcakit-harness-agent: production-grade Harness Agent on top of LangChain Deep Agents."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from harness_agent._version import __version__
from harness_agent.config import (
    HarnessAgentConfig,
    MediaGenerationConfig,
    ModelConfig,
    ProviderConfig,
)
from harness_agent.request import ChatRequest

if TYPE_CHECKING:
    from harness_agent.agent import HarnessAgent
    from harness_agent.init import InitResult, init_workspace
    from harness_agent.manager import AgentEntry, HarnessAgentManager
    from harness_agent.observability.logging import (
        current_log_file,
        default_log_dir,
        setup_logging,
        teardown_logging,
    )
    from harness_agent.protocols.langgraph import AgentEventType
    from harness_agent.providers import ModelPreset, ProviderPreset

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentEntry": ("harness_agent.manager", "AgentEntry"),
    "HarnessAgentManager": ("harness_agent.manager", "HarnessAgentManager"),
    "ModelPreset": ("harness_agent.providers", "ModelPreset"),
    "ProviderPreset": ("harness_agent.providers", "ProviderPreset"),
    "HarnessAgent": ("harness_agent.agent", "HarnessAgent"),
    "InitResult": ("harness_agent.init", "InitResult"),
    "init_workspace": ("harness_agent.init", "init_workspace"),
    "current_log_file": ("harness_agent.observability.logging", "current_log_file"),
    "default_log_dir": ("harness_agent.observability.logging", "default_log_dir"),
    "setup_logging": ("harness_agent.observability.logging", "setup_logging"),
    "teardown_logging": ("harness_agent.observability.logging", "teardown_logging"),
    "ChatProtocol": ("harness_agent.protocols", "ChatProtocol"),
    "register_protocol": ("harness_agent.protocols", "register_protocol"),
    "resolve_protocol": ("harness_agent.protocols", "resolve_protocol"),
    "AgentEventType": ("harness_agent.protocols.langgraph", "AgentEventType"),
    "SecurityPolicy": ("harness_agent.security.models", "SecurityPolicy"),
}


def __getattr__(name: str) -> object:  # pragma: no cover - thin re-export shim
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    return getattr(import_module(module_name), attr)


__all__ = [
    "AgentEntry",
    "AgentEventType",
    "ChatProtocol",
    "ChatRequest",
    "HarnessAgent",
    "HarnessAgentConfig",
    "HarnessAgentManager",
    "InitResult",
    "MediaGenerationConfig",
    "ModelConfig",
    "ModelPreset",
    "ProviderConfig",
    "ProviderPreset",
    "SecurityPolicy",
    "__version__",
    "current_log_file",
    "default_log_dir",
    "init_workspace",
    "register_protocol",
    "resolve_protocol",
    "setup_logging",
    "teardown_logging",
]
