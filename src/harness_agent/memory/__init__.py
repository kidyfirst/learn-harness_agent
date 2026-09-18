"""Memory adapter helpers for harness-agent.

The public surface of this subpackage is :class:`MemoryRuntime`, which wires
:class:`harness_memory.Memory` / :class:`harness_memory.MemoryService` into the
agent.

:class:`harness_agent.memory.llm_client.HarnessAgentLLMClient` is an internal
adapter used by :class:`MemoryRuntime` to expose the agent's
:class:`~harness_agent.llm.factory.ChatModelFactory` to harness-memory's
extractor / promotion / page-regeneration paths via the
:class:`harness_memory.ports.llm.LLMClient` protocol. It is intentionally not
re-exported here: callers who need a custom LLM client should implement the
``LLMClient`` protocol directly rather than subclassing or constructing this
adapter themselves.
"""

from __future__ import annotations

from harness_agent.memory.runtime import MemoryRuntime

__all__ = ["MemoryRuntime"]
