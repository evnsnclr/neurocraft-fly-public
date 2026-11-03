"""Explicit bounded handoffs used by the asynchronous bridge."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class LatestValueQueue(Generic[T]):
    """One-slot queue that deterministically coalesces an unread older value."""

    _queue: asyncio.Queue[T] = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    coalesced_count: int = 0

    def put_latest(self, value: T) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self.coalesced_count += 1
        self._queue.put_nowait(value)

    async def get(self) -> T:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    @property
    def size(self) -> int:
        return self._queue.qsize()


@dataclass(slots=True)
class BoundedCommandQueue(Generic[T]):
    """At-most-two pending commands; overflow is a fail-closed condition."""

    _queue: asyncio.Queue[T] = field(default_factory=lambda: asyncio.Queue(maxsize=2))
    overrun_count: int = 0

    def put(self, value: T) -> bool:
        if self._queue.full():
            self.overrun_count += 1
            return False
        self._queue.put_nowait(value)
        return True

    async def get(self) -> T:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    @property
    def size(self) -> int:
        return self._queue.qsize()
