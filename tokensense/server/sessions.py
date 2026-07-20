"""Per-session state for server transports.

The SDK's Project class owns one active sub-conversation per project; the
server instead needs one per *session*, because two sessions can hit the same
server process concurrently (possibly in different projects). Each session
owns its own sliding window and raw-turn history, backed by the same store
and middleware the SDK uses.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..memory.store import ProjectRecord, SubConversationRecord
from ..middleware import SlidingWindow


@dataclass
class ServerSession:
    session_id: str
    project: ProjectRecord
    sub_conversation: SubConversationRecord
    window: SlidingWindow
    raw_turns: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)
    # Serializes turns within one session; concurrency across sessions is the
    # normal case and needs no coordination beyond the store.
    lock: threading.Lock = field(default_factory=threading.Lock)

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def idle_for(self) -> float:
        return time.monotonic() - self.last_active

    @property
    def has_turns(self) -> bool:
        return bool(self.raw_turns)
