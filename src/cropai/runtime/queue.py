"""Bounded in-process inference queue. Reject when full; do not grow unbounded."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class BoundedInferenceQueue:
    def __init__(self, maxsize: int = 8, policy_when_full: str = "reject") -> None:
        self.maxsize = max(1, int(maxsize))
        self.policy = policy_when_full
        self._q: deque[Any] = deque()
        self.rejected = 0
        self.accepted = 0
        self.max_depth = 0

    def __len__(self) -> int:
        return len(self._q)

    def submit(self, item: T) -> bool:
        if len(self._q) >= self.maxsize:
            if self.policy == "drop_oldest" and self._q:
                self._q.popleft()
            else:
                self.rejected += 1
                return False
        self._q.append(item)
        self.accepted += 1
        self.max_depth = max(self.max_depth, len(self._q))
        return True

    def drain(self, fn: Callable[[T], Any]) -> list[Any]:
        out: list[Any] = []
        while self._q:
            item = self._q.popleft()
            out.append(fn(item))
        return out
