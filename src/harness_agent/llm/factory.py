"""Build a ``BaseChatModel`` from a ``ProviderConfig`` + ``ModelConfig`` pair.

Per design doc §3.3:
    - ``protocol="openai"``    → ``ChatOpenAI`` (also covers OpenAI-compatible
      vendors via ``base_url``).
    - ``protocol="anthropic"`` → ``ChatAnthropic``.
    - ``protocol="bedrock"``   → ``ChatBedrockConverse`` (optional dependency).

Instances are cached so that we only construct each ``(provider, model_id)``
combination once per agent. The cache key is provider-key-aware so two
``ProviderConfig`` objects sharing a base_url do not collide.

Reasoning-mode ``reasoning_content`` round-trip
-----------------------------------------------
``langchain-openai`` 1.x explicitly drops the non-standard ``reasoning_content``
field that providers like DeepSeek emit on assistant messages (see the
``ChatOpenAI`` docstring upstream). DeepSeek's "thinking" models then 400 on
the next request with::

    The `reasoning_content` in the thinking mode must be passed back to the API.

We solve that with a small ``ChatOpenAI`` subclass that captures
``reasoning_content`` into ``AIMessage.additional_kwargs`` on parse and
re-attaches it on serialize. The behaviour is no-op for providers that don't
emit the field, so it's always-on for ``protocol="openai"``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from harness_agent.config import ModelConfig, ProviderConfig
from harness_agent.llm.session_header import (
    aclose_session_http_clients,
    attach_session_http_clients,
    bind_anthropic_session_header,
    close_session_http_clients,
    session_http_clients,
)
from harness_agent.middleware.turn_model import resolve_turn_model_ref

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from harness_agent.config import HarnessAgentConfig
    from harness_agent.llm.access import ModelAccess
    from harness_agent.protocols.base import ChatProtocol


class ChatModelFactory:
    """Lazily builds and caches ``BaseChatModel`` instances.

    Designed to be held as a single instance on ``HarnessAgent`` (or shared
    across agents via ``AgentManager``) so that ``ModelRouterMiddleware`` can
    request models by ref ``"<provider_id>/<model_id>"`` without paying
    construction cost on every model call.

    Args:
        providers: List of :class:`~harness_agent.config.ProviderConfig`
            instances. Each provider's ``id`` is used as the lookup key for
            model refs of the form ``"<provider_id>/<model_id>"``.
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        agent_config: HarnessAgentConfig | None = None,
        get_protocol: Callable[[str | None], ChatProtocol] | None = None,
    ) -> None:
        # Build id→ProviderConfig index for O(1) lookups in get().
        self._providers: dict[str, ProviderConfig] = {p.id: p for p in providers}
        self._cache: dict[str, BaseChatModel] = {}
        self._agent_config = agent_config
        self._get_protocol = get_protocol

    def bind_runtime(
        self,
        *,
        get_protocol: Callable[[str | None], ChatProtocol],
        agent_config: HarnessAgentConfig | None = None,
    ) -> None:
        """Attach graph protocol access after :class:`HarnessAgent` compiles."""
        self._get_protocol = get_protocol
        if agent_config is not None:
            self._agent_config = agent_config

    def provider_configs(self) -> list[ProviderConfig]:
        """Snapshot of provider configs backing this factory (for config sync)."""
        return list(self._providers.values())

    def resolve_ref(
        self,
        spec: Any,
        *,
        messages: Sequence[Any] | None = None,
        configurable: Mapping[str, Any] | None = None,
    ) -> str:
        """Resolve a model spec to ``provider/model`` ref."""
        if isinstance(spec, str):
            stripped = spec.strip()
            if not stripped:
                raise ValueError("model spec must not be empty")
            lower = stripped.lower()
            if lower == "auto":
                return self.resolve_for_turn(
                    configurable=dict(configurable or {}),
                    messages=list(messages or []),
                )
            if "/" in stripped:
                return stripped
            bindings = (self._agent_config.role_bindings or {}) if self._agent_config else {}
            bound = bindings.get(stripped)
            if bound:
                return bound.strip()
            if self._agent_config is not None:
                return self._agent_config.pick_default_model_ref()
            raise ValueError(f"unknown model role {stripped!r} and no agent config for fallback")
        raise ValueError(f"unsupported model spec: {spec!r}")

    def resolve_for_turn(
        self,
        *,
        pick_default_ref: Callable[[], str] | None = None,
        pick_multimodal_ref: Callable[[], str | None] | None = None,
        configurable: Mapping[str, Any] | None = None,
        messages: Sequence[Any] | None = None,
        user_selector: Any = None,
        state: Any = None,
        runtime_config: Mapping[str, Any] | None = None,
    ) -> str:
        cfg = self._agent_config
        return resolve_turn_model_ref(
            pick_default_ref=pick_default_ref or (cfg.pick_default_model_ref if cfg is not None else lambda: ""),
            pick_multimodal_ref=pick_multimodal_ref or (cfg.pick_multimodal_model_ref if cfg is not None else None),
            configurable=configurable,
            messages=messages,
            user_selector=user_selector,
            state=state,
            runtime_config=runtime_config,
        )

    def list_refs(self, *, enabled_only: bool = True) -> list[str]:
        refs: list[str] = []
        for provider in self._providers.values():
            for model in provider.models:
                if enabled_only and not model.enabled:
                    continue
                refs.append(f"{provider.id}/{model.id}")
        return refs

    def get(
        self,
        model_ref: str,
        *,
        get_protocol: Callable[[str | None], ChatProtocol] | None = None,
    ) -> ModelAccess:
        """Return a :class:`ModelAccess` handle for ``provider/model``."""
        from harness_agent.llm.access import ModelAccess

        proto = get_protocol or self._get_protocol
        if proto is None:
            raise RuntimeError(
                "ChatModelFactory.get requires get_protocol=... or prior bind_runtime()",
            )
        return ModelAccess(
            model_ref=model_ref,
            chat_model=self.get_chat_model(model_ref),
            get_protocol=proto,
        )

    def get_for(
        self,
        spec: Any,
        *,
        messages: Sequence[Any] | None = None,
        configurable: Mapping[str, Any] | None = None,
        get_protocol: Callable[[str | None], ChatProtocol] | None = None,
        **kwargs: Any,
    ) -> ModelAccess:
        ref = self.resolve_ref(spec, messages=messages, configurable=configurable)
        return self.get(ref, get_protocol=get_protocol)

    def get_chat_model(self, model_ref: str) -> BaseChatModel:
        """Return the cached ``BaseChatModel`` (legacy escape hatch)."""
        cached = self._cache.get(model_ref)
        if cached is not None:
            return cached
        provider_key, _, model_id = model_ref.partition("/")
        provider = self._providers.get(provider_key)
        if provider is None:
            raise ValueError(f"Unknown provider {provider_key!r} in model ref {model_ref!r}")
        model = provider.get_model(model_id)
        if model is None or not model.enabled:
            raise ValueError(f"Model {model_ref!r} not found or disabled")
        instance = self._build(provider, model)
        self._cache[model_ref] = instance
        return instance

    def pop_cached(self, model_ref: str) -> None:
        """Drop a cached model and close any dedicated session-header clients."""
        instance = self._cache.pop(model_ref, None)
        if instance is not None:
            close_session_http_clients(instance)

    def close(self) -> None:
        """Close dedicated session-header clients and drop the model cache."""
        for instance in list(self._cache.values()):
            close_session_http_clients(instance)
        self._cache.clear()

    async def aclose(self) -> None:
        """Async variant of :meth:`close`."""
        for instance in list(self._cache.values()):
            await aclose_session_http_clients(instance)
        self._cache.clear()

    # ------------------------------------------------------------------
    # Construction by protocol
    # ------------------------------------------------------------------

    def _build(self, provider: ProviderConfig, model: ModelConfig) -> BaseChatModel:
        if provider.protocol == "openai":
            return _build_openai(provider, model)
        if provider.protocol == "anthropic":
            return _build_anthropic(provider, model)
        if provider.protocol == "bedrock":
            return _build_bedrock(provider, model)
        raise ValueError(f"Unsupported provider protocol: {provider.protocol!r}")


def build_chat_model(provider: ProviderConfig, model: ModelConfig) -> BaseChatModel:
    """Module-level convenience wrapper (mostly for tests)."""
    if provider.protocol == "openai":
        return _build_openai(provider, model)
    if provider.protocol == "anthropic":
        return _build_anthropic(provider, model)
    if provider.protocol == "bedrock":
        return _build_bedrock(provider, model)
    raise ValueError(f"Unsupported provider protocol: {provider.protocol!r}")


# ---------------------------------------------------------------------------
# Per-protocol builders (each isolates its own optional import)
# ---------------------------------------------------------------------------


def _build_openai(provider: ProviderConfig, model: ModelConfig) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - covered by install path
        raise ImportError(
            "langchain-openai is required for OpenAI-compatible providers. Install with: pip install langchain-openai",
        ) from exc

    cls = _get_reasoning_aware_chat_openai(ChatOpenAI)

    kwargs: dict[str, Any] = {
        "model": model.id,
        "base_url": provider.base_url,
        "api_key": provider.api_key,
        # langchain-openai only defaults stream_usage on for the official
        # OpenAI endpoint; with a custom base_url (always our case) it stays
        # off and streamed AIMessages carry no usage_metadata.
        "stream_usage": provider.stream_usage,
    }
    if model.native_tool_search:
        # OpenAI hosted tool search is a Responses API capability. Keep
        # Chat Completions as the default for OpenAI-compatible endpoints
        # that do not explicitly advertise it.
        kwargs["use_responses_api"] = True
        kwargs["output_version"] = "responses/v1"
    if provider.headers:
        kwargs["default_headers"] = dict(provider.headers)
    session_clients = None
    if provider.session_header:
        # Dedicated clients so the hook is not installed on langchain's shared
        # default httpx client (which other providers reuse).
        session_clients = session_http_clients(provider.session_header)
        kwargs["http_client"], kwargs["http_async_client"] = session_clients
    instance: BaseChatModel = cls(**kwargs)
    if session_clients is not None:
        attach_session_http_clients(instance, session_clients[0], session_clients[1])
    # Attach supported input modalities so the subclass can strip
    # unsupported content types (e.g. image_url for text-only models).
    instance._harness_supported_input = set(model.input)  # type: ignore[attr-defined]
    _stamp_native_tool_search_capability(instance, provider.protocol, model.native_tool_search)
    _inject_model_token_limits(instance, model)
    return instance


# ---------------------------------------------------------------------------
# reasoning_content round-trip subclass
# ---------------------------------------------------------------------------

# Built lazily on first use so importing harness_agent.llm.factory does not
# pay the ``langchain_openai`` import cost when only Anthropic/Bedrock are used.
# Keyed on the base class identity so test patches that swap ``ChatOpenAI`` for
# a mock don't poison the cache for subsequent real-class lookups.
_REASONING_AWARE_CHAT_OPENAI: dict[int, type[Any]] = {}


def _get_reasoning_aware_chat_openai(base_cls: type[Any]) -> type[Any]:
    """Return a ``ChatOpenAI`` subclass that round-trips ``reasoning_content``.

    Uses module-level memoisation keyed on the base class; the subclass is
    built once per (process, base class) pair.

    The subclass works by:

    * **Parse**: after ``super()._create_chat_result(...)`` runs, walk the
      raw ``choices`` array and copy each ``reasoning_content`` field into
      the matching ``AIMessage.additional_kwargs``.
    * **Serialize**: after ``super()._get_request_payload(...)`` runs, walk
      the assistant entries in ``payload["messages"]`` (chat/completions
      shape) and re-attach ``reasoning_content`` from the source
      ``AIMessage.additional_kwargs``.

    No module-level helpers are mutated, so the subclass is safe under
    concurrent use.
    """
    cached = _REASONING_AWARE_CHAT_OPENAI.get(id(base_cls))
    if cached is not None:
        return cached

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

    class _ReasoningAwareChatOpenAI(base_cls):  # type: ignore[misc]
        """ChatOpenAI subclass that round-trips ``reasoning_content``.

        Required for thinking-mode models from DeepSeek and similar
        providers that demand the field be echoed back on every turn.
        Behaviour is a no-op for providers that don't emit
        ``reasoning_content``.

        Three override points cover both streaming and non-streaming:

        * **_create_chat_result** — non-streaming parse: captures
          ``reasoning_content`` from the full response into
          ``AIMessage.additional_kwargs``.
        * **_convert_chunk_to_generation_chunk** — streaming parse:
          captures ``reasoning_content`` from each streamed delta into
          ``AIMessageChunk.additional_kwargs`` so it accumulates into the
          final message.
        * **_get_request_payload** — serialize: re-attaches
          ``reasoning_content`` from ``additional_kwargs`` into the
          outgoing request payload for the next turn.
        """

        def _create_chat_result(self, response: Any, generation_info: dict[str, Any] | None = None) -> Any:
            result = super()._create_chat_result(response, generation_info)

            # Extract reasoning_content from the raw response.
            # We try multiple access strategies because:
            #   1. model_dump() on some openai SDK versions drops extra fields
            #      from nested Pydantic models.
            #   2. The response may already be a dict (pre-processed by
            #      langchain-openai internals).
            #   3. Attribute access on the Pydantic model is the most reliable
            #      path when the response is an SDK object.

            if isinstance(response, dict):
                choices = response.get("choices") or []
                for gen, choice in zip(result.generations, choices, strict=False):
                    msg_dict = choice.get("message") if isinstance(choice, dict) else None
                    if not isinstance(msg_dict, dict):
                        continue
                    rc = msg_dict.get("reasoning_content")
                    if not rc:
                        continue
                    message = getattr(gen, "message", None)
                    if isinstance(message, AIMessage):
                        message.additional_kwargs.setdefault("reasoning_content", rc)
            else:
                # response is an SDK object (e.g. ChatCompletion) — use
                # attribute access which reliably surfaces extra fields.
                raw_choices = getattr(response, "choices", None) or []
                for gen, choice in zip(result.generations, raw_choices, strict=False):
                    msg = getattr(choice, "message", None)
                    if msg is None:
                        continue
                    # Try direct attribute (Pydantic extra field)
                    rc = getattr(msg, "reasoning_content", None)
                    if rc is None:
                        # Fallback: some SDK versions expose extras via
                        # model_extra or __pydantic_extra__
                        extras = getattr(msg, "model_extra", None) or {}
                        rc = extras.get("reasoning_content")
                    if not rc:
                        continue
                    message = getattr(gen, "message", None)
                    if isinstance(message, AIMessage):
                        message.additional_kwargs.setdefault("reasoning_content", rc)
            return result

        def _convert_chunk_to_generation_chunk(
            self,
            chunk: dict[str, Any],
            default_chunk_class: type[Any],
            base_generation_info: dict[str, Any] | None,
        ) -> Any:
            """Inject ``reasoning_content`` from streamed deltas into AIMessageChunk.

            In streaming mode, ``_create_chat_result`` is never called.
            Instead, each SSE chunk passes through this method.  The
            upstream implementation (``BaseChatOpenAI``) delegates to
            ``_convert_delta_to_message_chunk`` which silently drops the
            non-standard ``reasoning_content`` field from the delta dict.

            We call super() to get the default ``ChatGenerationChunk``,
            then patch ``reasoning_content`` from the raw delta onto the
            message's ``additional_kwargs`` so it accumulates into the
            final ``AIMessage`` that LangGraph stores in state.
            """
            gen_chunk = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
            if gen_chunk is None:
                return None

            # Extract reasoning_content from the raw chunk's delta.
            choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
            if not choices:
                return gen_chunk

            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if not isinstance(delta, dict):
                return gen_chunk

            rc = delta.get("reasoning_content")
            if not rc:
                return gen_chunk

            message = getattr(gen_chunk, "message", None)
            if isinstance(message, AIMessageChunk):
                message.additional_kwargs.setdefault("reasoning_content", "")
                message.additional_kwargs["reasoning_content"] += rc

            return gen_chunk

        def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            # ----------------------------------------------------------
            # Sanitize legacy image blocks BEFORE LangChain converts them.
            # Old tool results may contain Anthropic-style
            # {"type": "image", "source": {"type": "url", ...}} blocks
            # that LangChain's block translator can't handle.
            # Must run before super()._get_request_payload which triggers
            # _convert_message_to_dict → convert_to_openai_data_block.
            # ----------------------------------------------------------
            input_ = _sanitize_input_messages(input_)

            payload: dict[str, Any] = super()._get_request_payload(input_, stop=stop, **kwargs)

            # Only chat/completions has a flat ``messages`` array we can
            # walk; the responses API uses a different shape and these
            # providers don't expose it.
            messages_out = payload.get("messages")
            if not isinstance(messages_out, list):
                return payload

            # ----------------------------------------------------------
            # Strip unsupported content types for text-only models.
            # Models that don't declare "image" in their input modalities
            # will 400 on image_url / image blocks in conversation history.
            # ----------------------------------------------------------
            supported: set[str] = getattr(self, "_harness_supported_input", set()) or {"text", "image"}
            if "image" not in supported:
                _strip_unsupported_content_types(messages_out)

            # ----------------------------------------------------------
            # reasoning_content round-trip
            # ----------------------------------------------------------

            # Resolve source messages robustly.  _convert_input can fail
            # for exotic LanguageModelInput types (Runnables, template
            # partials, etc.), so we try multiple strategies.
            source_messages: list[BaseMessage] = []
            if isinstance(input_, list) and all(isinstance(m, BaseMessage) for m in input_):
                # Most common path in agent loops: input_ is already a
                # list of BaseMessage objects — use it directly.
                source_messages = input_
            else:
                try:
                    source_messages = self._convert_input(input_).to_messages()
                except (ValueError, TypeError, AttributeError, KeyError):
                    # Last resort: scan kwargs for a 'messages' key that
                    # some LangChain internals pass through.
                    msgs = kwargs.get("messages")
                    if isinstance(msgs, list):
                        source_messages = [m for m in msgs if isinstance(m, BaseMessage)]
                    if not source_messages:
                        return payload

            ai_index = 0
            for msg_dict in messages_out:
                if not isinstance(msg_dict, dict):
                    continue
                if msg_dict.get("role") != "assistant":
                    continue
                # Locate the matching AIMessage in source_messages.
                while ai_index < len(source_messages) and not isinstance(source_messages[ai_index], AIMessage):
                    ai_index += 1
                if ai_index >= len(source_messages):
                    break
                src = source_messages[ai_index]
                ai_index += 1
                rc = src.additional_kwargs.get("reasoning_content") if isinstance(src, AIMessage) else None
                if rc:
                    msg_dict["reasoning_content"] = rc
            return payload

    _REASONING_AWARE_CHAT_OPENAI[id(base_cls)] = _ReasoningAwareChatOpenAI
    return _ReasoningAwareChatOpenAI


# ---------------------------------------------------------------------------
# Legacy image block sanitization
# ---------------------------------------------------------------------------


def _sanitize_input_messages(input_: Any) -> Any:
    """Sanitize BaseMessage content before LangChain's block translator runs.

    Processes the input in-place when it's a list of BaseMessage objects.
    Converts Anthropic-style ``{"type": "image", "source": {...}}`` blocks
    (which LangChain can't handle) to the v1 standard format.
    """
    if not isinstance(input_, list):
        return input_
    from langchain_core.messages import BaseMessage

    for msg in input_:
        if not isinstance(msg, BaseMessage):
            continue
        content = msg.content
        if not isinstance(content, list):
            continue
        _sanitize_content_blocks(content)
    return input_


def _sanitize_content_blocks(content: list[Any]) -> None:
    """Fix legacy image blocks in a content block list in-place."""
    import base64 as _b64
    from pathlib import Path as _Path
    from urllib.parse import unquote as _unquote
    from urllib.parse import urlparse as _urlparse

    for i, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") != "image":
            continue
        source = block.get("source")
        if not isinstance(source, dict):
            continue
        # Blocks with top-level `base64` are already v1-compatible
        if "base64" in block:
            continue

        src_type = source.get("type", "")
        if src_type == "base64":
            data = source.get("data", "")
            mime = source.get("media_type", "") or "application/octet-stream"
            content[i] = {"type": "image", "base64": data, "mime_type": mime}
        elif src_type == "url":
            url = source.get("url", "")
            mime = source.get("media_type", "") or "image/png"
            if url.startswith("file://"):
                parsed = _urlparse(url)
                local_path = _Path(_unquote(parsed.path))
                if local_path.is_file():
                    try:
                        raw = local_path.read_bytes()
                        content[i] = {
                            "type": "image",
                            "base64": _b64.b64encode(raw).decode(),
                            "mime_type": mime,
                        }
                        continue
                    except OSError:
                        pass
            # Can't resolve — replace with text placeholder
            filename = block.get("filename", "") or url.rsplit("/", 1)[-1] or "image"
            content[i] = {"type": "text", "text": f"[image: {filename}]"}
        else:
            filename = block.get("filename", "") or "image"
            content[i] = {"type": "text", "text": f"[image: {filename}]"}


# ---------------------------------------------------------------------------
# Content-type filtering for text-only models
# ---------------------------------------------------------------------------

# Block types that require vision/multimodal support.
_IMAGE_CONTENT_TYPES: frozenset[str] = frozenset({"image_url", "image", "input_image"})


def _strip_unsupported_content_types(messages: list[Any]) -> None:
    """Remove image content blocks in-place from messages for text-only models.

    When conversation history contains ``image_url`` blocks (e.g. from a
    prior turn handled by a multimodal model) and the current model only
    supports text, sending those blocks causes a 400::

        unknown variant image_url, expected text

    This function:
    - Converts multipart content lists to text-only (drops image blocks,
      keeps text blocks).
    - If all content blocks are images, replaces with ``"[image]"`` so the
      message isn't empty.
    - Leaves string content and non-list content untouched.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        # Filter: keep only text blocks.
        text_parts: list[Any] = []
        had_image = False
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype in _IMAGE_CONTENT_TYPES:
                    had_image = True
                    continue
            # Keep text blocks and any other unknown types.
            text_parts.append(block)

        if not had_image:
            # No images found — nothing to do for this message.
            continue

        if not text_parts:
            # All blocks were images — replace with placeholder.
            msg["content"] = "[image]"
        elif len(text_parts) == 1 and isinstance(text_parts[0], dict) and text_parts[0].get("type") == "text":
            # Single text block remaining — collapse to a plain string
            # for maximum compatibility.
            msg["content"] = text_parts[0].get("text", "")
        else:
            msg["content"] = text_parts


def _inject_model_token_limits(instance: BaseChatModel, model: ModelConfig) -> None:
    """Inject model context metadata into the LangChain profile.

    DeepAgents' ``SummarizationMiddleware`` reads ``model.profile["max_input_tokens"]``
    to switch from conservative fixed-count fallbacks to fraction-based thresholds
    (85% trigger, 10% keep).  LangChain only populates ``.profile`` for OpenAI
    first-party models; for all other providers it is ``None``.

    When ``max_input_tokens > 0``, we set the value directly so that all models
    — including DeepSeek, Qwen, Hunyuan, etc. — benefit from accurate context
    management.  When the model already has a profile with ``max_input_tokens``
    (i.e. an OpenAI first-party model), the harness value *overwrites* it so
    that manually configured limits take precedence.
    """
    max_input_tokens = model.max_input_tokens
    context_window = model.effective_context_window
    max_output_tokens = model.max_output_tokens
    if max_input_tokens <= 0 and context_window <= 0 and max_output_tokens <= 0:
        return
    profile = getattr(instance, "profile", None)
    if not isinstance(profile, dict):
        # profile is None or some unexpected type — replace with a minimal dict.
        with contextlib.suppress(AttributeError, TypeError):
            object.__setattr__(instance, "profile", {})
        profile = getattr(instance, "profile", None)
    if not isinstance(profile, dict):
        return
    if max_input_tokens > 0:
        profile["max_input_tokens"] = max_input_tokens
    if context_window > 0:
        profile["context_window"] = context_window
    if max_output_tokens > 0:
        profile["max_output_tokens"] = max_output_tokens
    # Keep immutable baselines outside the mutable profile. Per-request output
    # reservations may temporarily lower profile["max_input_tokens"].
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(instance, "_harness_max_input_tokens", max_input_tokens)
        object.__setattr__(instance, "_harness_context_window", context_window)
        object.__setattr__(instance, "_harness_max_output_tokens", max_output_tokens)


def _stamp_native_tool_search_capability(
    instance: BaseChatModel,
    provider: str,
    enabled: bool,
) -> None:
    """Attach Harness-only capability metadata to a routed model instance."""
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(instance, "_harness_native_tool_search", enabled)
        object.__setattr__(instance, "_harness_tool_search_provider", provider)


def _build_anthropic(provider: ProviderConfig, model: ModelConfig) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langchain-anthropic is required for the anthropic protocol. Install with: pip install langchain-anthropic",
        ) from exc

    kwargs: dict[str, Any] = {
        "model": model.id,
        "api_key": provider.api_key,
    }
    # ChatAnthropic accepts ``base_url`` for proxy/relay configurations.
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    if provider.headers:
        kwargs["default_headers"] = dict(provider.headers)
    # Enable extended thinking when model.thinking is explicitly True.
    if model.thinking is True:
        kwargs["thinking"] = {"type": "adaptive"}
    instance_a: BaseChatModel = ChatAnthropic(**kwargs)
    if provider.session_header:
        bind_anthropic_session_header(instance_a, provider.session_header)
    _stamp_native_tool_search_capability(instance_a, provider.protocol, model.native_tool_search)
    _inject_model_token_limits(instance_a, model)
    return instance_a


def _build_bedrock(provider: ProviderConfig, model: ModelConfig) -> BaseChatModel:
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:
        raise ImportError(
            "langchain-aws is required for the bedrock protocol. "
            "Install with: pip install 'orcakit-harness-agent[bedrock]'",
        ) from exc

    # Bedrock auth is typically AWS-credential-driven; api_key/base_url here
    # are accepted for symmetry but not all combinations are meaningful.
    kwargs: dict[str, Any] = {"model": model.id}
    if provider.headers:
        kwargs["additional_model_request_fields"] = dict(provider.headers)
    instance_b: BaseChatModel = ChatBedrockConverse(**kwargs)
    _inject_model_token_limits(instance_b, model)
    return instance_b


__all__ = ["ChatModelFactory", "ModelAccess", "build_chat_model"]
