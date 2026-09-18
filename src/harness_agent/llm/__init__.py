"""LLM subpackage: provider configuration → LangChain ``BaseChatModel``."""

from __future__ import annotations

from harness_agent.llm.factory import build_chat_model

__all__ = ["build_chat_model"]
