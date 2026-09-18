"""Teams: optional inbox-driven agent-to-agent collaboration."""

from __future__ import annotations

from harness_agent.teams.inbox import (
    HarnessAgentInboxManager,
    InboxMessage,
    InboxStatus,
    PeerResult,
)
from harness_agent.teams.processor import (
    ReplyEvent,
    ReplyStatus,
    TeamProcessor,
    default_compose_followup,
)
from harness_agent.teams.team_manager import TeamManager
from harness_agent.teams.tools import AskAgentInput, build_team_tools
from harness_agent.teams.util import (
    PeerCall,
    PeerSession,
    build_one_shot_request,
    derive_peer_thread_id,
)

__all__ = [
    "AskAgentInput",
    "HarnessAgentInboxManager",
    "InboxMessage",
    "InboxStatus",
    "PeerCall",
    "PeerResult",
    "PeerSession",
    "ReplyEvent",
    "ReplyStatus",
    "TeamManager",
    "TeamProcessor",
    "build_one_shot_request",
    "build_team_tools",
    "default_compose_followup",
    "derive_peer_thread_id",
]
