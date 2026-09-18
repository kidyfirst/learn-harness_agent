"""Global inbox queue for asynchronous agent-to-agent collaboration.

One :class:`HarnessAgentInboxManager` serves all agents in a shared
:class:`~harness_agent.registry.AgentRegistry`. A single background worker
consumes messages serially:

    registry.get(target).agent.call(message) -> compose_followup ->
    registry.get(source).agent.call(prompt) -> on_reply
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from harness_agent.backends.utils import ASYNC_DEFENSIVE_OP_ERRORS
from harness_agent.messages import extract_call_response
from harness_agent.request import ChatRequest
from harness_agent.teams.processor import ReplyEvent, TeamProcessor
from harness_agent.teams.util import build_one_shot_request, derive_peer_thread_id

logger = logging.getLogger(__name__)

InboxStatus = Literal["queued", "running", "replying", "done", "failed", "cancelled"]
_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})


@dataclass
class InboxMessage:
    """A queued cross-agent request and its routing/status info."""

    id: str
    target_agent_id: str
    source_agent_id: str
    source_thread_id: str | None
    message: str
    user_id: str | int
    status: InboxStatus = "queued"
    original_user_prompt: str | None = None
    error_text: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PeerResult:
    """Return value of a peer interaction (sync call or async submit)."""

    mode: Literal["sync", "background"]
    agent_id: str
    name: str
    job_id: str | None = None
    thread_id: str | None = None
    status: str | None = None
    response: str | None = None
    message: str | None = None


class HarnessAgentInboxManager:
    """Process-wide inbox: enqueue messages, a serial worker fulfils them."""

    def __init__(
        self,
        *,
        call_agent: Callable[[str, ChatRequest], Awaitable[dict[str, Any]]],
        processor: TeamProcessor,
        invoke_target: Callable[[InboxMessage], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self._call_agent = call_agent
        self._processor = processor
        self._invoke_target = invoke_target
        self._messages: dict[str, InboxMessage] = {}
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._in_flight: str | None = None

    # ------------------------------------------------------------------
    # Public queue API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        *,
        target_agent_id: str,
        source_agent_id: str,
        source_thread_id: str | None,
        message: str,
        user_id: str | int,
        original_user_prompt: str | None = None,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a message to the inbox and return its id (returns immediately)."""
        self.start()
        mid = job_id or uuid.uuid4().hex
        self._messages[mid] = InboxMessage(
            id=mid,
            target_agent_id=target_agent_id,
            source_agent_id=source_agent_id,
            source_thread_id=source_thread_id,
            message=message,
            user_id=user_id,
            original_user_prompt=original_user_prompt,
            metadata=dict(metadata or {}),
        )
        self._queue.put_nowait(mid)
        return mid

    def get(self, inbox_id: str) -> InboxMessage | None:
        return self._messages.get(inbox_id)

    def list(
        self,
        *,
        target: str | None = None,
        source: str | None = None,
        status: InboxStatus | None = None,
    ) -> list[InboxMessage]:
        return [
            m
            for m in self._messages.values()
            if (target is None or m.target_agent_id == target)
            and (source is None or m.source_agent_id == source)
            and (status is None or m.status == status)
        ]

    def cancel(self, inbox_id: str) -> bool:
        msg = self._messages.get(inbox_id)
        if msg is None or msg.status in _TERMINAL_STATUSES:
            return False
        self._set_status(msg, "cancelled")
        if self._in_flight == inbox_id:
            # Worker checks status between phases; no task cancellation needed.
            return True
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop(), name="harness-inbox")

    async def shutdown(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._worker, timeout=30.0)
        except TimeoutError:
            self._worker.cancel()
        self._worker = None

    def cancel_worker(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
        self._worker = None

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        while True:
            mid = await self._queue.get()
            try:
                if mid is None:
                    break
                msg = self._messages.get(mid)
                if msg is None or msg.status == "cancelled":
                    continue
                await self._process(msg)
            except ASYNC_DEFENSIVE_OP_ERRORS:
                logger.exception("inbox message failed id=%s", mid)
            finally:
                self._queue.task_done()

    def _is_cancelled(self, msg: InboxMessage) -> bool:
        return msg.status == "cancelled"

    async def _process(self, msg: InboxMessage) -> None:
        self._in_flight = msg.id
        try:
            if self._is_cancelled(msg):
                return
            self._set_status(msg, "running")
            result_text: str | None = None
            error_text: str | None = None
            status: Literal["done", "failed", "cancelled"] = "done"
            try:
                if self._invoke_target is not None:
                    result = await self._invoke_target(msg)
                else:
                    sk_raw = msg.metadata.get("session_key")
                    session_key = sk_raw if isinstance(sk_raw, str) and sk_raw.strip() else None
                    target_req = build_one_shot_request(
                        user_id=msg.user_id,
                        agent_id=msg.target_agent_id,
                        text=msg.message,
                        source="inbox",
                        thread_id=(
                            derive_peer_thread_id(msg.source_thread_id, msg.target_agent_id)
                            if msg.source_thread_id
                            else None
                        ),
                        session_key=session_key,
                    )
                    result = await self._call_agent(msg.target_agent_id, target_req)
                result_text = extract_call_response(result) if isinstance(result, dict) else ""
            except ASYNC_DEFENSIVE_OP_ERRORS as exc:
                error_text = str(exc)
                status = "failed"

            if self._is_cancelled(msg):
                return

            self._set_status(msg, "replying")
            reply_text = await self._synthesize_reply(msg, result_text, error_text)
            if self._is_cancelled(msg):
                return
            if reply_text is None and status == "done":
                status = "failed"
                error_text = error_text or "source agent reply failed"

            msg.error_text = error_text
            self._set_status(msg, status)
            await self._processor.on_reply(
                ReplyEvent(
                    inbox_id=msg.id,
                    status=status,
                    source_agent_id=msg.source_agent_id,
                    source_thread_id=msg.source_thread_id,
                    target_agent_id=msg.target_agent_id,
                    user_id=msg.user_id,
                    reply_text=reply_text,
                    error_text=error_text,
                    metadata=msg.metadata,
                )
            )
            self._prune_terminal(msg.id)
        finally:
            self._in_flight = None

    async def _synthesize_reply(
        self,
        msg: InboxMessage,
        result_text: str | None,
        error_text: str | None,
    ) -> str | None:
        if self._is_cancelled(msg):
            return None
        prompt = self._processor.compose_followup(msg, result_text=result_text, error_text=error_text)
        try:
            source_req = build_one_shot_request(
                user_id=msg.user_id,
                agent_id=msg.source_agent_id,
                text=prompt,
                source="inbox",
                thread_id=msg.source_thread_id,
            )
            reply = await self._call_agent(msg.source_agent_id, source_req)
            return extract_call_response(reply) if isinstance(reply, dict) else ""
        except ASYNC_DEFENSIVE_OP_ERRORS:
            logger.exception("inbox source reply failed id=%s", msg.id)
            return None

    def _set_status(self, msg: InboxMessage, status: InboxStatus) -> None:
        msg.status = status
        msg.updated_at = datetime.now(tz=UTC)

    def _prune_terminal(self, inbox_id: str) -> None:
        msg = self._messages.get(inbox_id)
        if msg is not None and msg.status in _TERMINAL_STATUSES:
            self._messages.pop(inbox_id, None)


__all__ = [
    "HarnessAgentInboxManager",
    "InboxMessage",
    "InboxStatus",
    "PeerResult",
]
