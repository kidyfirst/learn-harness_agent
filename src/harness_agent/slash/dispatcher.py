"""Backward-compat re-export — prefer ``from harness_agent.slash import ...``."""

from harness_agent.slash.core import (
    RuntimeSlashDispatcher,
    build_runtime_dispatcher,
)

__all__ = ["RuntimeSlashDispatcher", "build_runtime_dispatcher"]
