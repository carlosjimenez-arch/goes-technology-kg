"""Replay-first gateway; online acquisition is an explicit, separately authorized mode."""

import threading

from goes_tech_kg.llm.replay import ReplayMiss, ReplayStore
from goes_tech_kg.llm.vertex import VertexClient
from goes_tech_kg.schemas.llm import LLMRequest, ResponseRecord


class LLMGateway:
    def __init__(self, store: ReplayStore, online: VertexClient | None = None):
        self.store = store
        self.online = online
        self.acquired = 0
        self.replayed = 0
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest, user_content: str) -> ResponseRecord:
        record = self.store.get(request)
        if record is not None:
            with self._lock:
                self.replayed += 1
            return record
        if self.online is None:
            raise ReplayMiss(f"no recorded response for request {request.key}")
        record = self.online.generate(request, user_content)
        self.store.put(record)
        with self._lock:
            self.acquired += 1
        return record
