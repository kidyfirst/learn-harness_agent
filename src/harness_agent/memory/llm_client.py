"""``HarnessAgentLLMClient`` — adapt the agent's models to harness-memory.

``harness-memory``'s extractor / promotion / page-regen paths talk to the
host model through the :class:`harness_memory.ports.llm.LLMClient` protocol. This
module wraps the agent's existing :class:`ChatModelFactory` so we can reuse
the same providers and credentials the rest of the agent already uses.

Model selection
---------------
Extract uses **one** model, not a light/heavy pair. Resolution order:

1. optional aux override (``aux_model``, or the legacy light/heavy aliases)
2. the live chat model (:meth:`set_current_model`)
3. the agent's ``default_model``

``tier`` only changes the timeout. If a ref fails (expired subscription,
400, missing model), ``call_llm`` tries the next one in that list.

Failure discipline
------------------
Any transport / model error is wrapped in :class:`LLMClientError` so the
plugin's extractor / promotion worker can degrade gracefully (write the
``failure_reason`` and move on) — never bubble into the user reply path.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Literal

from harness_memory.ports.llm import LLMClientError, LLMTier
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from harness_agent.llm.factory import ChatModelFactory

logger = logging.getLogger(__name__)

# Vendored from octop's ``infra.utils.llm_text.strip_thinking`` — harness-agent
# is published standalone and must not import the host Octop package, so this
# stays a duplicate rather than a shared import. Keep the two in sync by hand
# if the stripping rule changes.
_THINKING_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


class HarnessAgentLLMClient:
    """Adapter from :class:`ChatModelFactory` to :class:`LLMClient`.

    Args:
        factory: The agent's model factory; used to build / cache
            ``BaseChatModel`` instances by ``"<provider>/<model>"`` ref.
        aux_model: Optional extract override. When unset, extract follows
            the live chat model and then ``default_model``.
        light_model / heavy_model: Legacy aliases for ``aux_model``. The
            first non-empty value wins; both tiers share it.
        default_model: Last-resort model ref when no aux and no live chat
            model have been recorded yet.
    """

    # Default timeouts in seconds. Light covers per-session extraction and alias
    # disambiguation; heavy covers cross-session consolidation. Hard caps prevent
    # upstream gateway stalls or slow inference from blocking extract threads.
    DEFAULT_LIGHT_TIMEOUT_S: float = 120.0
    DEFAULT_HEAVY_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        factory: ChatModelFactory,
        *,
        aux_model: str | None = None,
        light_model: str | None = None,
        heavy_model: str | None = None,
        default_model: str | None = None,
        light_timeout_s: float | None = None,
        heavy_timeout_s: float | None = None,
    ) -> None:
        configured = _first_ref(aux_model, light_model, heavy_model)
        if configured is None and default_model is None:
            raise ValueError(
                "HarnessAgentLLMClient requires aux_model or default_model",
            )
        self._factory = factory
        self._aux_model = configured
        self._default_model = default_model
        self._light_timeout_s = light_timeout_s if light_timeout_s is not None else self.DEFAULT_LIGHT_TIMEOUT_S
        self._heavy_timeout_s = heavy_timeout_s if heavy_timeout_s is not None else self.DEFAULT_HEAVY_TIMEOUT_S
        self._current_lock = threading.Lock()
        self._current_model: str | None = None

    def set_current_model(self, ref: str | None) -> None:
        """Record the model the user is actually chatting with.

        When no aux override is set, extract uses this ref. When aux is
        set and fails, extract retries here.
        """
        value = ref.strip() if isinstance(ref, str) and ref.strip() else None
        with self._current_lock:
            self._current_model = value

    # ------------------------------------------------------------------
    # LLMClient protocol
    # ------------------------------------------------------------------

    def call_llm(
        self,
        prompt: str,
        *,
        tier: LLMTier = "light",
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        refs = self._candidate_refs()
        if not refs:
            raise LLMClientError(f"no model configured for tier={tier!r}")

        last_exc: BaseException | None = None
        for index, ref in enumerate(refs):
            try:
                return self._invoke_ref(
                    ref,
                    prompt,
                    tier=tier,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            except Exception as exc:
                last_exc = exc
                next_ref = refs[index + 1] if index + 1 < len(refs) else None
                if next_ref is None:
                    break
                logger.info(
                    "HarnessAgentLLMClient: %s failed for tier=%s; falling back to %s: %s",
                    ref,
                    tier,
                    next_ref,
                    exc,
                )

        assert last_exc is not None
        failed_ref = refs[-1]
        raise LLMClientError(
            f"chat model {failed_ref!r} failed for tier={tier!r}: {last_exc}",
        ) from last_exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _candidate_refs(self) -> list[str]:
        """Return distinct refs: aux override, then live chat, then default."""
        with self._current_lock:
            current = self._current_model
        ordered: list[str] = []
        seen: set[str] = set()
        for ref in (self._aux_model, current, self._default_model):
            stripped = _first_ref(ref)
            if stripped is None or stripped in seen:
                continue
            seen.add(stripped)
            ordered.append(stripped)
        return ordered

    def _invoke_ref(
        self,
        ref: str,
        prompt: str,
        *,
        tier: LLMTier,
        system: str | None,
        max_tokens: int | None,
        temperature: float | None,
        response_format: Literal["text", "json"],
    ) -> str:
        try:
            model = self._factory.get_chat_model(ref)
        except Exception as exc:
            raise LLMClientError(
                f"failed to build chat model {ref!r} for tier={tier!r}: {exc}",
            ) from exc

        bound_model = self._apply_call_options(
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            timeout_s=self._light_timeout_s if tier == "light" else self._heavy_timeout_s,
        )

        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        try:
            response = bound_model.invoke(messages)
        except Exception as exc:
            # Provider 400s (expired subscription, bad request) are not
            # always RuntimeError/OSError subclasses. Treat every model
            # failure as degrade-able so extract never kills the user turn.
            logger.info(
                "HarnessAgentLLMClient.call_llm failed (model=%s, tier=%s): %s",
                ref,
                tier,
                exc,
            )
            raise LLMClientError(
                f"chat model {ref!r} failed for tier={tier!r}: {exc}",
            ) from exc

        return _stringify(response.content)

    def _apply_call_options(
        self,
        model: object,
        *,
        max_tokens: int | None,
        temperature: float | None,
        response_format: Literal["text", "json"],
        timeout_s: float | None = None,
    ) -> Any:
        """Bind per-call kwargs the underlying ``BaseChatModel`` understands.

        We use ``bind()`` (not mutation) so the cached model instance in
        :class:`ChatModelFactory` keeps clean defaults across callers.
        Unsupported kwargs are silently ignored by ``bind()``; we do not
        try to whitelist them per provider — too brittle.
        """
        kwargs: dict[str, object] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if timeout_s is not None and timeout_s > 0:
            # LangChain's OpenAI / Anthropic adapters both accept ``timeout``.
            # bind() also tolerates unknown kwargs, so provider allowlisting is not needed.
            kwargs["timeout"] = timeout_s
            # Do not add ``max_retries=0`` here. Some providers, such as Tencent
            # hy3-preview's ``Completions.parse()``, reject that kwarg with
            # TypeError before sending the request. The timeout already bounds
            # the worst case for background extraction stalls.
        # ``response_format={"type": "json_object"}`` is OpenAI-specific.
        # Anthropic's LangChain adapter tries to convert it to its own
        # ``output_config.format`` shape and rejects ``json_object`` before
        # the request is sent, so only pass this hint to OpenAI-style models.
        # Other providers still receive the JSON prompt and the memory layer
        # validates/parses the returned string.
        if response_format == "json" and _supports_openai_response_format(model):
            kwargs["response_format"] = {"type": "json_object"}

        if not kwargs:
            return model
        bind = getattr(model, "bind", None)
        if bind is None:  # pragma: no cover - all langchain chat models expose bind
            return model
        try:
            return bind(**kwargs)
        except (TypeError, ValueError, AttributeError):  # pragma: no cover - defensive against provider-specific quirks
            logger.debug(
                "HarnessAgentLLMClient: model.bind(**%s) rejected; falling back to defaults",
                kwargs,
            )
            return model


def _first_ref(*refs: str | None) -> str | None:
    for ref in refs:
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    return None


def _supports_openai_response_format(model: object) -> bool:
    """Return whether ``model`` accepts OpenAI-style ``response_format``."""
    for cls in type(model).mro():
        module = getattr(cls, "__module__", "")
        if module.startswith("langchain_openai"):
            return True
    return False


def _stringify(content: object) -> str:
    """Reduce a ``BaseMessage.content`` (str | list[block]) to plain text.

    Some providers (notably local/open-weight models routed through an
    OpenAI-compat gateway) emit reasoning either as a dedicated
    ``type: "thinking"``/``"reasoning"`` block, or inline as literal
    ``<think>...</think>`` text within an otherwise plain string. Both are
    stripped here so the extractor's JSON parser never has to deal with
    reasoning traces leaking into the response it parses.
    """
    if isinstance(content, list):
        joined = "".join(_stringify_block(block) for block in content)
        return _THINKING_RE.sub("", joined).strip()
    text = content if isinstance(content, str) else str(content)
    return _THINKING_RE.sub("", text).strip()


def _stringify_block(block: object) -> str:
    """Extract text from one content block, or ``""`` to skip it.

    Skips ``type: "thinking"``/``"reasoning"`` blocks and non-text blocks
    (image refs, tool calls, …) — the extractor's prompts always ask for
    plain text.
    """
    if not isinstance(block, dict):
        return str(block)
    block_type = str(block.get("type") or "").lower()
    if block_type in ("thinking", "reasoning"):
        return ""
    text = block.get("text")
    return text if isinstance(text, str) else ""


__all__ = ["HarnessAgentLLMClient"]
