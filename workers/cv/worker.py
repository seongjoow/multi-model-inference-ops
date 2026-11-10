"""CV 워커: Redis 큐에서 작업을 꺼내 마이크로배치로 추론하고, 결과를 Redis로 돌려준다.

실행: python -m workers.cv.worker
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import redis
from prometheus_client import start_http_server

from common import keys
from common.jsonlog import log_event, setup_logging
from workers.cv import metrics as m
from workers.cv.backends import make_backend
from workers.cv.queue import make_queue
from workers.cv.settings import Settings

log = logging.getLogger("cv-worker")


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


@dataclass
class Job:
    id: str
    image: bytes
    enqueued_at: float  # 게이트웨이가 넣은 시각 (unix time)
    request_id: str
    popped_at: float  # 워커가 꺼낸 시각 (unix time)
    attempts: int = 0


class Worker:
    def __init__(self, s: Settings, r: redis.Redis) -> None:
        self.s = s
        self.r = r
        self.queue = make_queue(s.queue_mode, r, s.worker_id)
        self.backend = make_backend(s)
        self.stopping = False
        self._last_beat = 0.0

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        recovered = self.queue.recover()
        if recovered:
            m.RECOVERED.inc(recovered)
            log_event(log, "recovered_jobs", logging.WARNING, count=recovered, worker=self.s.worker_id)
        log_event(
            log,
            "worker_started",
            worker=self.s.worker_id,
            queue_mode=self.s.queue_mode,
            backend=self.backend.name,
            batch_size=self.s.batch_size,
            max_wait_ms=self.s.max_wait_ms,
        )
        while not self.stopping:
            try:
                self._heartbeat()
                batch = self._collect_batch()
                if batch:
                    self._process(batch)
            except redis.ConnectionError as exc:
                log_event(log, "redis_unavailable", logging.ERROR, error=repr(exc))
                time.sleep(1)
        log_event(log, "worker_stopped", worker=self.s.worker_id)

    def _heartbeat(self) -> None:
        now = time.time()
        if now - self._last_beat < 1.0:
            return
        self._last_beat = now
        self.r.set(keys.heartbeat(self.s.worker_id), repr(now), ex=self.s.heartbeat_ttl_s)
        Path(self.s.heartbeat_file).touch()
        m.LAST_HEARTBEAT.set(now)

    def _collect_batch(self) -> list[tuple[str, float]]:
        """첫 작업은 최대 1초 블로킹(그 사이 heartbeat 갱신), 이후 max_wait_ms 안에서 batch_size까지 채운다.

        배치 창은 워커 쪽에서 짧게 폴링해 채운다. Redis의 블로킹 timeout은 한가한 서버에서
        약 100ms 단위(hz=10)로만 처리되어, BLPOP timeout=0.01 이 실제로는 100ms를 기다린다.
        """
        first = self.queue.take(timeout=1.0)
        if first is None:
            return []
        batch = [(first, time.time())]
        deadline = time.monotonic() + self.s.max_wait_ms / 1000
        while len(batch) < self.s.batch_size:
            job_id = self.queue.take_nowait()
            if job_id is not None:
                batch.append((job_id, time.time()))
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.001))
        return batch

    # ------------------------------------------------------------------ processing

    def _process(self, batch: list[tuple[str, float]]) -> None:
        t_start = time.time()

        pipe = self.r.pipeline(transaction=False)
        for job_id, _ in batch:
            pipe.hmget(keys.job(job_id), "image", "enqueued_at", "request_id", "deadline")
        rows = pipe.execute()

        jobs: list[Job] = []
        missing: list[str] = []
        expired: list[tuple[str, str]] = []
        for (job_id, popped_at), (image, enqueued_at, request_id, deadline) in zip(batch, rows):
            if image is None:  # TTL 만료 등으로 본문이 없는 작업
                missing.append(job_id)
                continue
            if deadline is not None and t_start > float(deadline):
                # 게이트웨이가 이미 시간 초과(504)로 응답한 작업. 추론해도 받을 사람이 없으니 건너뛴다
                expired.append((job_id, (request_id or b"").decode()))
                continue
            jobs.append(Job(job_id, image, float(enqueued_at), (request_id or b"").decode(), popped_at))
        if missing:
            self.queue.ack(missing)
            m.JOBS.labels("missing").inc(len(missing))
            log_event(log, "jobs_missing", logging.WARNING, job_ids=missing)
        if expired:
            pipe = self.r.pipeline(transaction=False)
            for job_id, _ in expired:
                pipe.hset(keys.job(job_id), mapping={"status": "expired", "finished_at": repr(time.time())})
                pipe.hdel(keys.job(job_id), "image")
            pipe.execute()
            self.queue.ack([job_id for job_id, _ in expired])
            m.JOBS.labels("expired").inc(len(expired))
            log_event(log, "jobs_expired", logging.WARNING, count=len(expired),
                      request_ids=[rid for _, rid in expired][:20])

        pipe = self.r.pipeline(transaction=False)
        for job in jobs:
            pipe.hincrby(keys.job(job.id), "attempts", 1)
        for job, attempts in zip(jobs, pipe.execute()):
            job.attempts = int(attempts)

        runnable: list[Job] = []
        for job in jobs:
            if self.queue.supports_retry and job.attempts > self.s.max_attempts:
                self._fail(job, "max_attempts_exceeded", f"attempts={job.attempts}", to_dlq=True)
            else:
                runnable.append(job)

        t_dec = time.perf_counter()
        images: list[Any] = []
        valid: list[Job] = []
        for job in runnable:
            try:
                images.append(self.backend.decode(job.image))
                valid.append(job)
            except Exception as exc:
                self._fail(job, "decode_error", str(exc))
        decode_s = time.perf_counter() - t_dec
        if not valid:
            return

        t_inf = time.perf_counter()
        try:
            outputs = self.backend.infer(images)
        except Exception as exc:
            log.exception("inference_failed", extra={"fields": {"job_ids": [j.id for j in valid]}})
            if self.queue.supports_retry:
                self.queue.nack([j.id for j in valid])  # 재시도 (상한을 넘으면 DLQ)
                m.JOBS.labels("retried").inc(len(valid))
            else:
                for job in valid:
                    self._fail(job, "inference_error", repr(exc))
            return
        infer_s = time.perf_counter() - t_inf
        t_done = time.time()

        pipe = self.r.pipeline(transaction=False)
        queue_waits = []
        for job, detections in zip(valid, outputs):
            queue_wait = job.popped_at - job.enqueued_at
            batch_wait = t_start - job.popped_at
            queue_waits.append(queue_wait)
            m.QUEUE_WAIT.observe(max(queue_wait, 0))
            m.BATCH_WAIT.observe(max(batch_wait, 0))
            self._deliver(
                pipe,
                job,
                {
                    "job_id": job.id,
                    "request_id": job.request_id,
                    "status": "done",
                    "worker": self.s.worker_id,
                    "model": self.backend.name,
                    "num_detections": len(detections),
                    "detections": detections,
                    "timing": {
                        "queue_wait_ms": _ms(queue_wait),
                        "batch_wait_ms": _ms(batch_wait),
                        "decode_ms": _ms(decode_s),
                        "inference_ms": _ms(infer_s),
                        "batch_size": len(valid),
                        "worker_total_ms": _ms(t_done - job.popped_at),
                        "attempts": job.attempts,
                    },
                },
            )
        pipe.execute()
        self.queue.ack([j.id for j in valid])

        m.JOBS.labels("done").inc(len(valid))
        m.BATCH_SIZE.observe(len(valid))
        m.DECODE.observe(decode_s)
        m.INFERENCE.observe(infer_s)
        log_event(
            log,
            "batch_done",
            batch_size=len(valid),
            decode_ms=_ms(decode_s),
            inference_ms=_ms(infer_s),
            queue_wait_ms_max=_ms(max(queue_waits)),
            request_ids=[j.request_id for j in valid],
        )

    def _deliver(self, pipe: Any, job: Job, payload: dict[str, Any]) -> None:
        data = json.dumps(payload)
        pipe.hset(keys.job(job.id), mapping={"status": payload["status"], "result": data, "finished_at": repr(time.time())})
        pipe.hdel(keys.job(job.id), "image")
        pipe.rpush(keys.result(job.id), data)
        pipe.expire(keys.result(job.id), self.s.result_ttl_s)

    def _fail(self, job: Job, code: str, detail: str, to_dlq: bool = False) -> None:
        pipe = self.r.pipeline(transaction=False)
        self._deliver(
            pipe,
            job,
            {
                "job_id": job.id,
                "request_id": job.request_id,
                "status": "failed",
                "error_code": code,
                "error": detail,
                "worker": self.s.worker_id,
                "timing": {"attempts": job.attempts},
            },
        )
        if to_dlq:
            pipe.rpush(keys.DLQ, job.id)
        pipe.execute()
        self.queue.ack([job.id])
        m.JOBS.labels(code).inc()
        log_event(log, "job_failed", logging.WARNING, job_id=job.id, request_id=job.request_id, error_code=code, error=detail)


def main() -> None:
    s = Settings.from_env()
    setup_logging("cv-worker", s.log_level)
    r = redis.Redis.from_url(s.redis_url)
    for attempt in range(30):  # Redis가 먼저 떠 있지 않아도 잠시 기다린다
        try:
            r.ping()
            break
        except redis.ConnectionError:
            time.sleep(1)
    else:
        raise SystemExit("redis is not reachable")

    start_http_server(s.metrics_port)
    worker = Worker(s, r)

    def _stop(signum: int, _frame: Any) -> None:
        log_event(log, "stop_requested", signal=signal.Signals(signum).name)
        worker.stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    worker.run()


if __name__ == "__main__":
    main()
