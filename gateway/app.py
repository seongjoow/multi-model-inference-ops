"""API Gateway.

- LLM: OpenAI 호환 /v1/chat/completions 로 프록시. 항상 upstream 스트리밍으로 받아 TTFT·토큰 수를 잰다.
  게이트웨이 단에서 동시 요청 수를 제한한다 (LLM_MAX_INFLIGHT). 단일 프로세스로 띄워야 상한이 전역으로 맞는다.
- CV: 이미지를 Redis 큐에 넣고 job_id를 돌려준다. wait=true 면 결과까지 기다렸다가 돌려준다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from common import keys
from common.jsonlog import log_event, setup_logging
from gateway import metrics as m
from gateway.settings import Settings

settings = Settings.from_env()
setup_logging("gateway", settings.log_level)
log = logging.getLogger("gateway")


CV_WAIT_MAX_S = 600  # wait=true 요청이 정할 수 있는 최대 대기 (timeout 쿼리 상한과 같음)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex


# --------------------------------------------------------------------------- lifecycle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 일반 명령(LLEN, 작업 등록 등)용. redis-py 8.x 기본값대로 소켓 읽기 한도 5초.
    pool = aioredis.BlockingConnectionPool.from_url(
        settings.redis_url, max_connections=settings.redis_max_connections, timeout=5
    )
    app.state.redis = aioredis.Redis(connection_pool=pool)
    # 결과 대기(BLPOP) 전용. redis-py 8.x는 socket_timeout 기본값이 5초라, 결과를 최대 CV_WAIT_TIMEOUT_S 동안
    # 기다려야 하는 BLPOP이 5초에 TimeoutError로 끊긴다 (D3 본 실험에서 CV 과부하 때 나온 500의 원인).
    # 그래서 기다리는 연결은 읽기 한도를 최대 대기 시간보다 길게 둔다.
    wait_pool = aioredis.BlockingConnectionPool.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        timeout=5,
        socket_timeout=CV_WAIT_MAX_S + 5,
        socket_connect_timeout=5,
    )
    app.state.redis_wait = aioredis.Redis(connection_pool=wait_pool)
    app.state.http = httpx.AsyncClient(
        base_url=settings.llm_base_url,
        timeout=httpx.Timeout(settings.llm_request_timeout_s, connect=5.0),
        limits=httpx.Limits(max_connections=settings.llm_max_inflight + 16),
    )
    app.state.llm_sem = asyncio.Semaphore(settings.llm_max_inflight)
    poller = asyncio.create_task(_poll_queue_stats(app.state.redis))
    log_event(
        log,
        "gateway_started",
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_max_inflight=settings.llm_max_inflight,
        cv_max_queue=settings.cv_max_queue,
    )
    try:
        yield
    finally:
        poller.cancel()
        await app.state.http.aclose()
        await app.state.redis.aclose()
        await app.state.redis_wait.aclose()


app = FastAPI(title="multi-model-inference-ops gateway", lifespan=lifespan)


@app.exception_handler(RedisConnectionError)
@app.exception_handler(RedisTimeoutError)
async def _redis_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """Redis 연결을 못 얻거나 응답이 늦으면(연결 풀 고갈, Redis 장애·지연) 500 대신 503 + Retry-After.

    처리되지 않은 예외는 500이 되어 "서버 버그"와 구분되지 않는다. 일시적인 과부하·장애이므로
    "잠시 후 다시"라는 뜻의 503으로 돌려주고, 로그(redis_unavailable)와 지표에서 따로 셀 수 있게 한다.
    """
    rid = _request_id(request)
    workload = "cv" if request.url.path.startswith("/v1/cv") else "llm" if request.url.path.startswith("/v1/llm") else "other"
    m.REQUESTS.labels(workload, "503").inc()
    log_event(log, "redis_unavailable", logging.WARNING, request_id=rid, path=request.url.path, error=repr(exc))
    resp = _json_error(503, "temporarily unavailable: redis connection", rid)
    resp.headers["Retry-After"] = "1"
    return resp


async def _poll_queue_stats(r: aioredis.Redis) -> None:
    """CV 큐 상태를 주기적으로 읽어 게이지로 노출한다. 워커가 죽어도 이 값은 계속 갱신된다."""
    healthy = True
    while True:
        try:
            depth = await r.llen(keys.QUEUE)
            dlq = await r.llen(keys.DLQ)
            processing = 0
            async for key in r.scan_iter(match=keys.PROCESSING_PATTERN):
                processing += await r.llen(key)
            workers = 0
            async for _ in r.scan_iter(match=keys.HEARTBEAT_PATTERN):
                workers += 1
            m.CV_QUEUE_DEPTH.set(depth)
            m.CV_DLQ_DEPTH.set(dlq)
            m.CV_PROCESSING.set(processing)
            m.CV_WORKERS_ALIVE.set(workers)
            if not healthy:
                log_event(log, "queue_stats_recovered")
            healthy = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Redis 장애 중에도 게이트웨이는 살아 있어야 한다
            if healthy:
                log_event(log, "queue_stats_failed", logging.WARNING, error=repr(exc))
            healthy = False
        await asyncio.sleep(settings.stats_interval_s)


# --------------------------------------------------------------------------- health / metrics


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"
    try:
        resp = await request.app.state.http.get("/v1/models", timeout=2.0)
        checks["llm"] = "ok" if resp.status_code == 200 else f"http {resp.status_code}"
    except httpx.HTTPError as exc:
        checks["llm"] = f"error: {type(exc).__name__}"
    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"ready": ready, "checks": checks, "llm_base_url": settings.llm_base_url},
        status_code=200 if ready else 503,
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------- LLM


@dataclass
class LlmCall:
    request_id: str
    stream: bool
    t0: float  # 게이트웨이 수신 시각 (perf_counter)
    queue_wait: float = 0.0
    t_send: float = 0.0
    t_first: float | None = None
    t_end: float | None = None
    chunks: int = 0
    usage: dict[str, Any] | None = None
    model: str | None = None
    finish_reason: str | None = None
    content: list[str] = field(default_factory=list)
    finished: bool = False

    def observe_chunk(self, data: str, keep_text: bool) -> None:
        try:
            obj = json.loads(data)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        self.model = self.model or obj.get("model")
        if obj.get("usage"):
            self.usage = obj["usage"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content")
            if text:
                if self.t_first is None:
                    self.t_first = time.perf_counter()
                self.chunks += 1
                if keep_text and delta.get("content"):
                    self.content.append(delta["content"])
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    @property
    def output_tokens(self) -> int:
        if self.usage and self.usage.get("completion_tokens") is not None:
            return int(self.usage["completion_tokens"])
        return self.chunks  # usage가 없으면 콘텐츠 청크 수로 근사

    def timing(self) -> dict[str, Any]:
        end = self.t_end or time.perf_counter()
        out: dict[str, Any] = {
            "queue_wait_ms": _ms(self.queue_wait),
            "e2e_ms": _ms(end - self.t0),
            "output_tokens": self.output_tokens,
        }
        if self.t_first is not None:
            out["ttft_ms"] = _ms(self.t_first - self.t0)
            out["upstream_ttft_ms"] = _ms(self.t_first - self.t_send)
            if self.output_tokens > 1:
                out["tpot_ms"] = _ms((end - self.t_first) / (self.output_tokens - 1))
        return out


def _record_llm(call: LlmCall, status: int, error: str | None = None) -> None:
    m.REQUESTS.labels("llm", str(status)).inc()
    timing = call.timing()
    if status == 200:
        m.LLM_E2E.observe(timing["e2e_ms"] / 1000)
        m.LLM_OUTPUT_TOKENS.inc(call.output_tokens)
        if "ttft_ms" in timing:
            m.LLM_TTFT.observe(timing["ttft_ms"] / 1000)
        if "tpot_ms" in timing:
            m.LLM_TPOT.observe(timing["tpot_ms"] / 1000)
    log_event(
        log,
        "llm_request",
        logging.INFO if status == 200 else logging.WARNING,
        request_id=call.request_id,
        status=status,
        stream=call.stream,
        prompt_tokens=(call.usage or {}).get("prompt_tokens"),
        finish_reason=call.finish_reason,
        error=error,
        **timing,
    )


def _finish_llm(call: LlmCall, status: int, error: str | None = None) -> None:
    """슬롯을 반납하고 지표·로그를 남긴다. 어떤 경로로 끝나도 정확히 한 번만 실행된다."""
    if call.finished:
        return
    call.finished = True
    if call.t_end is None:
        call.t_end = time.perf_counter()
    app.state.llm_sem.release()
    m.LLM_INFLIGHT.dec()
    _record_llm(call, status, error)


def _json_error(status: int, message: str, request_id: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        {"error": message, "request_id": request_id, **extra},
        status_code=status,
        headers={"X-Request-ID": request_id},
    )


@app.post("/v1/llm/chat")
async def llm_chat(request: Request) -> Response:
    t0 = time.perf_counter()
    rid = _request_id(request)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict) or not body.get("messages"):
        m.REQUESTS.labels("llm", "400").inc()
        return _json_error(400, "JSON body with 'messages' is required", rid)

    call = LlmCall(request_id=rid, stream=bool(body.get("stream")), t0=t0)
    upstream_body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    upstream_body.setdefault("model", settings.llm_model)

    # 1) 동시성 슬롯 확보 (admission control)
    m.LLM_WAITING.inc()
    try:
        await asyncio.wait_for(request.app.state.llm_sem.acquire(), timeout=settings.llm_queue_timeout_s)
    except asyncio.TimeoutError:
        call.queue_wait = time.perf_counter() - t0
        call.t_end = time.perf_counter()
        _record_llm(call, 503, "llm concurrency limit: queue timeout")
        return _json_error(503, "LLM is busy, retry later", rid, retry_after_s=1)
    finally:
        m.LLM_WAITING.dec()
    call.queue_wait = time.perf_counter() - t0
    m.LLM_QUEUE_WAIT.observe(call.queue_wait)
    m.LLM_INFLIGHT.inc()

    # 2) upstream 호출 (헤더까지 받은 뒤 스트리밍 여부 결정)
    client: httpx.AsyncClient = request.app.state.http
    call.t_send = time.perf_counter()
    try:
        upstream = await client.send(
            client.build_request("POST", "/v1/chat/completions", json=upstream_body), stream=True
        )
    except httpx.TimeoutException as exc:
        _finish_llm(call, 504, f"upstream timeout: {type(exc).__name__}")
        return _json_error(504, "LLM upstream timeout", rid)
    except httpx.HTTPError as exc:
        _finish_llm(call, 502, f"upstream unreachable: {type(exc).__name__}")
        return _json_error(502, "LLM upstream unreachable", rid)

    if upstream.status_code != 200:
        detail = (await upstream.aread()).decode(errors="replace")[:500]
        await upstream.aclose()
        _finish_llm(call, upstream.status_code, detail)
        return _json_error(upstream.status_code, "LLM upstream error", rid, detail=detail)

    if call.stream:
        return StreamingResponse(
            _relay_sse(call, upstream),
            media_type="text/event-stream",
            headers={"X-Request-ID": rid, "Cache-Control": "no-cache"},
        )
    return await _collect_completion(call, upstream)


async def _relay_sse(call: LlmCall, upstream: httpx.Response) -> AsyncIterator[str]:
    """upstream SSE를 그대로 넘기면서 토큰을 센다. [DONE] 직전에 게이트웨이 측정값을 SSE 주석으로 붙인다."""
    status, error = 200, None
    try:
        async for line in upstream.aiter_lines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    call.t_end = time.perf_counter()
                    yield f": gateway {json.dumps(call.timing())}\n"
                else:
                    call.observe_chunk(data, keep_text=False)
            yield line + "\n"
    except (asyncio.CancelledError, GeneratorExit):
        status, error = 499, "client disconnected"
        raise
    except httpx.HTTPError as exc:
        status, error = 502, f"upstream stream error: {type(exc).__name__}"
        yield f"data: {json.dumps({'error': error, 'request_id': call.request_id})}\n\n"
    finally:
        await upstream.aclose()
        _finish_llm(call, status, error)


async def _collect_completion(call: LlmCall, upstream: httpx.Response) -> JSONResponse:
    status, error = 200, None
    try:
        async for line in upstream.aiter_lines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    call.observe_chunk(data, keep_text=True)
    except httpx.HTTPError as exc:
        status, error = 502, f"upstream stream error: {type(exc).__name__}"
    finally:
        await upstream.aclose()
        _finish_llm(call, status, error)
    if status != 200:
        return _json_error(status, error or "upstream error", call.request_id)
    return JSONResponse(
        {
            "id": f"chatcmpl-{call.request_id}",
            "object": "chat.completion",
            "model": call.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(call.content)},
                    "finish_reason": call.finish_reason,
                }
            ],
            "usage": call.usage,
            "gateway": call.timing(),
        },
        headers={"X-Request-ID": call.request_id},
    )


# --------------------------------------------------------------------------- CV


@app.post("/v1/cv/detect")
async def cv_detect(
    request: Request,
    image: UploadFile = File(...),
    wait: bool = Query(False, description="true면 결과가 나올 때까지 기다렸다가 반환"),
    timeout: float | None = Query(None, gt=0, le=600, description="wait=true일 때 최대 대기 초"),
) -> JSONResponse:
    t0 = time.perf_counter()
    rid = _request_id(request)
    r: aioredis.Redis = request.app.state.redis

    depth = await r.llen(keys.QUEUE)
    if depth >= settings.cv_max_queue:
        m.REQUESTS.labels("cv", "429").inc()
        log_event(log, "cv_rejected", logging.WARNING, request_id=rid, queue_depth=depth)
        return _json_error(429, "CV queue is full", rid, queue_depth=depth)

    data = await image.read()
    if not data:
        m.REQUESTS.labels("cv", "400").inc()
        return _json_error(400, "empty image", rid)

    job_id = uuid.uuid4().hex
    wait_s = min(timeout or settings.cv_wait_timeout_s, CV_WAIT_MAX_S)
    now = time.time()
    fields: dict[str, Any] = {
        "image": data,
        "status": "queued",
        "enqueued_at": repr(now),
        "request_id": rid,
        "size": len(data),
    }
    if wait:
        # 이 시각이 지나면 게이트웨이는 이미 504를 돌려줬으므로 결과를 받을 사람이 없다.
        # 워커는 마감이 지난 작업을 추론하지 않고 건너뛴다 ('주인 없는 작업'에 GPU를 쓰지 않도록).
        fields["deadline"] = repr(now + wait_s)
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(keys.job(job_id), mapping=fields)
        pipe.expire(keys.job(job_id), settings.cv_job_ttl_s)
        pipe.rpush(keys.QUEUE, job_id)
        await pipe.execute()

    headers = {"X-Request-ID": rid}
    if not wait:
        m.REQUESTS.labels("cv", "202").inc()
        log_event(log, "cv_enqueued", request_id=rid, job_id=job_id, size=len(data), queue_depth=depth + 1)
        return JSONResponse({"job_id": job_id, "status": "queued", "request_id": rid}, status_code=202, headers=headers)

    popped = await request.app.state.redis_wait.blpop([keys.result(job_id)], timeout=wait_s)
    e2e = time.perf_counter() - t0
    if popped is None:
        m.REQUESTS.labels("cv", "504").inc()
        log_event(log, "cv_request", logging.WARNING, request_id=rid, job_id=job_id, status=504, e2e_ms=_ms(e2e))
        return JSONResponse(
            {"job_id": job_id, "status": "timeout", "request_id": rid}, status_code=504, headers=headers
        )

    payload = json.loads(popped[1])
    if payload.get("status") == "done":
        status = 200
        m.CV_E2E.observe(e2e)
    else:
        status = 422 if payload.get("error_code") == "decode_error" else 500
    payload["gateway"] = {"e2e_ms": _ms(e2e)}
    m.REQUESTS.labels("cv", str(status)).inc()
    timing = payload.get("timing", {})
    log_event(
        log,
        "cv_request",
        logging.INFO if status == 200 else logging.WARNING,
        request_id=rid,
        job_id=job_id,
        status=status,
        e2e_ms=_ms(e2e),
        queue_wait_ms=timing.get("queue_wait_ms"),
        batch_wait_ms=timing.get("batch_wait_ms"),
        inference_ms=timing.get("inference_ms"),
        batch_size=timing.get("batch_size"),
        error_code=payload.get("error_code"),
    )
    return JSONResponse(payload, status_code=status, headers=headers)


@app.get("/v1/cv/jobs/{job_id}")
async def cv_job(job_id: str, request: Request) -> JSONResponse:
    raw = await request.app.state.redis.hgetall(keys.job(job_id))
    if not raw:
        return JSONResponse({"job_id": job_id, "error": "not found or expired"}, status_code=404)
    info = {k.decode(): v.decode(errors="replace") for k, v in raw.items() if k != b"image"}
    result = json.loads(info.pop("result")) if "result" in info else None
    return JSONResponse({"job_id": job_id, **info, "result": result})
