"""부하 생성기.

두 가지 방식으로 요청을 보낸다.
- 고정 도착률(open loop, --rate): 응답과 상관없이 정해진 속도로 요청이 "도착"한다. 실제 사용자 트래픽과 같아서
  서버가 감당 못 하면 줄이 길어지는 게 그대로 드러난다. 경합·SLO 실험의 기본.
- 고정 동시성(closed loop, --concurrency): N명이 "응답 받으면 바로 다음 요청"을 반복한다. 최대 처리량을 볼 때 쓴다.

요청마다 한 줄씩 CSV에 남기고(requests.csv), 실험 조건(meta.json, env.txt)과 요약(summary.json)을 함께 저장한다.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import socket
import subprocess
import sys
import time
from itertools import count
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from loadgen.stats import format_table, percentile, summarize

argparse_ns = Any  # argparse.Namespace (타입 표기용)

FIELDS = [
    "run", "workload", "seq", "phase", "request_id", "t_send_unix", "send_lag_ms",
    "status", "error", "latency_ms", "client_ttft_ms", "output_tokens",
    "gw_queue_wait_ms", "gw_ttft_ms", "gw_upstream_ttft_ms", "gw_tpot_ms", "gw_e2e_ms",
    "cv_queue_wait_ms", "cv_batch_wait_ms", "cv_decode_ms", "cv_inference_ms", "cv_batch_size", "image",
]

# 프롬프트 본문. 단어 수로 입력 길이를 맞춘다.
BASE_TEXT = (
    "The harbor town woke slowly under a grey sky. Fishing boats rocked against the pier while gulls argued "
    "over scraps near the market stalls. An old ferry captain checked the tide tables twice, as he did every "
    "morning, and wrote the numbers in a notebook that had outlived three of his boats. Children ran past the "
    "bakery on their way to school, and the smell of bread followed them down the hill. By noon the wind had "
    "turned, the clouds broke apart, and the whole bay shone like hammered metal."
)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def build_prompt(seq: int, input_words: int) -> str:
    """요청마다 앞부분을 다르게 해 서버의 prefix cache가 측정을 왜곡하지 않게 한다."""
    words = BASE_TEXT.split()
    body = " ".join((words * (input_words // len(words) + 1))[:input_words])
    return f"Request {seq}. Continue the following text.\n\n{body}"


def load_images(directory: str, n: int, seed: int) -> list[tuple[str, bytes]]:
    paths = sorted(Path(directory).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"이미지가 없습니다: {directory} (COCO val2017을 data/ 에 풀었는지 확인)")
    rng = random.Random(seed)
    picked = rng.sample(paths, min(n, len(paths)))
    return [(p.name, p.read_bytes()) for p in picked]  # 디스크 I/O가 측정에 끼지 않게 미리 메모리에 올림


# --------------------------------------------------------------------------- 요청 1건


async def llm_request(client: httpx.AsyncClient, args: argparse_ns, seq: int, rid: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": build_prompt(seq, args.input_words)}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if args.fixed_output:
        body["ignore_eos"] = True  # 끝 토큰이 나와도 max_tokens까지 생성 → 출력 길이 고정 (vLLM 지원)
    row: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", "/v1/llm/chat", json=body, headers={"X-Request-ID": rid}) as resp:
            row["status"] = resp.status_code
            if resp.status_code != 200:
                row["error"] = (await resp.aread()).decode(errors="replace")[:200]
            else:
                chunks = 0
                async for line in resp.aiter_lines():
                    if line.startswith(": gateway "):  # 게이트웨이가 [DONE] 직전에 붙이는 측정값
                        g = json.loads(line[len(": gateway "):])
                        row.update(
                            gw_queue_wait_ms=g.get("queue_wait_ms"),
                            gw_ttft_ms=g.get("ttft_ms"),
                            gw_upstream_ttft_ms=g.get("upstream_ttft_ms"),
                            gw_tpot_ms=g.get("tpot_ms"),
                            gw_e2e_ms=g.get("e2e_ms"),
                            output_tokens=g.get("output_tokens"),
                        )
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        if isinstance(obj, dict) and obj.get("error"):
                            row["status"], row["error"] = "stream_error", str(obj["error"])[:200]
                            continue
                        for ch in obj.get("choices") or []:
                            if (ch.get("delta") or {}).get("content"):
                                if "client_ttft_ms" not in row:
                                    row["client_ttft_ms"] = _ms(time.perf_counter() - t0)
                                chunks += 1
                if row.get("output_tokens") is None:
                    row["output_tokens"] = chunks
    except httpx.TimeoutException:
        row["status"], row["error"] = "timeout", "client timeout"
    except httpx.HTTPError as exc:
        row["status"], row["error"] = "conn_error", f"{type(exc).__name__}: {exc}"[:200]
    row["latency_ms"] = _ms(time.perf_counter() - t0)
    return row


async def cv_request(
    client: httpx.AsyncClient, args: argparse_ns, seq: int, rid: str, images: list[tuple[str, bytes]]
) -> dict[str, Any]:
    name, data = images[seq % len(images)]
    row: dict[str, Any] = {"image": name}
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            "/v1/cv/detect",
            params={"wait": "true"},
            files={"image": (name, data, "image/jpeg")},
            headers={"X-Request-ID": rid},
        )
        row["status"] = resp.status_code
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if resp.status_code == 200:
            t = payload.get("timing", {})
            row.update(
                cv_queue_wait_ms=t.get("queue_wait_ms"),
                cv_batch_wait_ms=t.get("batch_wait_ms"),
                cv_decode_ms=t.get("decode_ms"),
                cv_inference_ms=t.get("inference_ms"),
                cv_batch_size=t.get("batch_size"),
                gw_e2e_ms=(payload.get("gateway") or {}).get("e2e_ms"),
            )
        else:
            row["error"] = str(payload.get("error") or resp.text)[:200]
    except httpx.TimeoutException:
        row["status"], row["error"] = "timeout", "client timeout"
    except httpx.HTTPError as exc:
        row["status"], row["error"] = "conn_error", f"{type(exc).__name__}: {exc}"[:200]
    row["latency_ms"] = _ms(time.perf_counter() - t0)
    return row


# --------------------------------------------------------------------------- 실행


def _git_commit() -> str:
    if os.getenv("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _copy_env(src: str, dst: Path) -> bool:
    """실험 당시 설정(.env)을 결과와 함께 남긴다. 비밀값으로 보이는 줄은 뺀다."""
    path = Path(src)
    if not path.is_file():
        return False
    secret = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    lines = [ln for ln in path.read_text().splitlines() if not any(s in ln.split("=")[0].upper() for s in secret)]
    dst.write_text("\n".join(lines) + "\n")
    return True


async def run(args: argparse_ns) -> int:
    run_id = args.name or f"{time.strftime('%Y%m%d-%H%M%S')}-{args.workload}"
    out = Path(args.out) if args.out else Path("results") / run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"결과 폴더가 이미 있습니다: {out} (--name 을 바꾸세요)")
    out.mkdir(parents=True, exist_ok=True)

    images = load_images(args.images, args.num_images, args.seed) if args.workload == "cv" else []
    send: Callable[[httpx.AsyncClient, int, str], Awaitable[dict[str, Any]]]
    if args.workload == "llm":
        send = lambda c, seq, rid: llm_request(c, args, seq, rid)  # noqa: E731
    else:
        send = lambda c, seq, rid: cv_request(c, args, seq, rid, images)  # noqa: E731

    mode = f"rate={args.rate}/s ({args.arrival})" if args.rate else f"concurrency={args.concurrency}"
    print(f"[loadgen] run={run_id} workload={args.workload} {mode} warmup={args.warmup}s duration={args.duration}s"
          f" target={args.target}", file=sys.stderr)

    meta: dict[str, Any] = {
        "run": run_id,
        "workload": args.workload,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "env_file_copied": _copy_env(args.env_file, out / "env.txt"),
    }

    f = open(out / "requests.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    rows: list[dict[str, Any]] = []
    stats = {"sent": 0, "done": 0, "ok": 0, "inflight": 0, "dropped": 0}

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        writer.writerow(row)
        stats["done"] += 1
        if str(row.get("status")) == "200":
            stats["ok"] += 1
        if stats["done"] % 50 == 0:
            f.flush()

    limits = httpx.Limits(max_connections=args.max_inflight, max_keepalive_connections=args.max_inflight)
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    start_wall = time.time()
    start = time.perf_counter()
    warmup_end = start + args.warmup
    end = warmup_end + args.duration
    meta.update(start_unix=start_wall, measure_start_unix=start_wall + args.warmup)

    async def one(client: httpx.AsyncClient, seq: int, scheduled: float) -> None:
        now = time.perf_counter()
        base = {
            "run": run_id,
            "workload": args.workload,
            "seq": seq,
            "phase": "warmup" if now < warmup_end else "measure",
            "request_id": f"{run_id}-{seq:06d}",
            "t_send_unix": round(time.time(), 4),
            "send_lag_ms": _ms(now - scheduled),  # 예정 시각보다 늦게 보낸 정도 (부하 생성기 자체 과부하 확인)
        }
        stats["sent"] += 1
        stats["inflight"] += 1
        try:
            row = await send(client, seq, base["request_id"])
        finally:
            stats["inflight"] -= 1
        record({**base, **row})

    async def progress() -> None:
        while True:
            await asyncio.sleep(5)
            elapsed = time.perf_counter() - start
            print(f"[loadgen] t={elapsed:5.0f}s sent={stats['sent']} done={stats['done']} ok={stats['ok']}"
                  f" inflight={stats['inflight']} dropped={stats['dropped']}", file=sys.stderr)

    reporter = asyncio.create_task(progress())
    interrupted = False
    try:
        async with httpx.AsyncClient(base_url=args.target, limits=limits, timeout=timeout) as client:
            if args.rate:
                rng = random.Random(args.seed)
                tasks: set[asyncio.Task] = set()
                next_t, seq = start, 0
                while next_t < end:
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if stats["inflight"] >= args.max_inflight:
                        stats["dropped"] += 1  # 부하 생성기 한도 초과: 보내지 못한 요청도 기록
                        record({"run": run_id, "workload": args.workload, "seq": seq,
                                "phase": "warmup" if time.perf_counter() < warmup_end else "measure",
                                "t_send_unix": round(time.time(), 4), "status": "dropped",
                                "error": f"loadgen max_inflight={args.max_inflight} reached"})
                    else:
                        task = asyncio.create_task(one(client, seq, next_t))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
                    seq += 1
                    next_t += rng.expovariate(args.rate) if args.arrival == "poisson" else 1 / args.rate
                if tasks:
                    print(f"[loadgen] 전송 종료, 남은 응답 {len(tasks)}건 대기", file=sys.stderr)
                    await asyncio.wait(tasks, timeout=args.timeout + 5)
            else:
                counter = count()

                async def user() -> None:
                    while time.perf_counter() < end:
                        await one(client, next(counter), time.perf_counter())

                await asyncio.gather(*(user() for _ in range(args.concurrency)))
    except (asyncio.CancelledError, KeyboardInterrupt):
        interrupted = True
    finally:
        reporter.cancel()
        f.close()

    end_wall = time.time()
    meta.update(end_unix=end_wall, interrupted=interrupted)
    summary = summarize(rows, duration_s=float(args.duration))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + format_table([(run_id, summary)]))
    lags = [float(r["send_lag_ms"]) for r in rows if r.get("send_lag_ms") not in (None, "")]
    if (percentile(lags, 99) or 0) > 50:
        print("\n[경고] 요청을 예정보다 50ms 이상 늦게 보낸 경우가 1% 넘게 있습니다. 부하 생성기 자체가 바빠"
              " 결과가 왜곡됐을 수 있습니다 (send_lag_ms 열 확인).")
    if stats["dropped"]:
        print(f"\n[경고] 부하 생성기 동시 한도 때문에 보내지 못한 요청 {stats['dropped']}건 (status=dropped).")
    grafana = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
    print(f"\n결과: {out}/  (requests.csv, summary.json, meta.json, env.txt)")
    print(f"Grafana: {grafana}/d/inference-ops/?from={int(start_wall * 1000)}&to={int(end_wall * 1000) + 5000}")
    return 130 if interrupted else 0
