"""CV 작업 큐.

simple   : BLPOP으로 꺼낸다. 처리 중 워커가 죽으면 그 작업은 사라진다 (장애 재현용 기준선).
reliable : BLMOVE로 cv:queue -> cv:processing:<worker> 로 옮겨 두고, 처리가 끝나면 지운다(ack).
           워커가 재시작되면 자기 처리 중 목록을 큐 앞쪽으로 되돌린다(recover).
"""

from __future__ import annotations

import redis

from common import keys


class SimpleQueue:
    supports_retry = False

    def __init__(self, r: redis.Redis) -> None:
        self.r = r

    def take(self, timeout: float) -> str | None:
        item = self.r.blpop([keys.QUEUE], timeout=timeout)
        return item[1].decode() if item else None

    def take_nowait(self) -> str | None:
        item = self.r.lpop(keys.QUEUE)
        return item.decode() if item else None

    def ack(self, job_ids: list[str]) -> None:
        pass

    def nack(self, job_ids: list[str]) -> None:
        raise NotImplementedError

    def recover(self) -> int:
        return 0


class ReliableQueue:
    supports_retry = True

    def __init__(self, r: redis.Redis, worker_id: str) -> None:
        self.r = r
        self.processing = keys.processing(worker_id)

    def take(self, timeout: float) -> str | None:
        item = self.r.blmove(keys.QUEUE, self.processing, timeout, "LEFT", "RIGHT")
        return item.decode() if item else None

    def take_nowait(self) -> str | None:
        item = self.r.lmove(keys.QUEUE, self.processing, "LEFT", "RIGHT")
        return item.decode() if item else None

    def ack(self, job_ids: list[str]) -> None:
        if not job_ids:
            return
        pipe = self.r.pipeline(transaction=False)
        for job_id in job_ids:
            pipe.lrem(self.processing, 1, job_id)
        pipe.execute()

    def nack(self, job_ids: list[str]) -> None:
        """다시 시도하도록 큐 뒤쪽으로 돌려보낸다."""
        pipe = self.r.pipeline(transaction=True)
        for job_id in job_ids:
            pipe.lrem(self.processing, 1, job_id)
            pipe.rpush(keys.QUEUE, job_id)
        pipe.execute()

    def recover(self) -> int:
        """이전 실행에서 ack하지 못한 작업을 큐 앞쪽으로 되돌린다 (원래 순서 유지)."""
        moved = 0
        while self.r.lmove(self.processing, keys.QUEUE, "RIGHT", "LEFT") is not None:
            moved += 1
        return moved


def make_queue(mode: str, r: redis.Redis, worker_id: str) -> SimpleQueue | ReliableQueue:
    if mode == "simple":
        return SimpleQueue(r)
    if mode == "reliable":
        return ReliableQueue(r, worker_id)
    raise ValueError(f"unknown CV_QUEUE_MODE: {mode!r} (simple | reliable)")
