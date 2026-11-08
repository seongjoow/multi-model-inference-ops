"""요약 통계. 외부 라이브러리 없이 백분위를 계산한다."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def percentile(values: list[float], q: float) -> float | None:
    """선형 보간 백분위 (q: 0~100). 값이 없으면 None."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _floats(rows: Iterable[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v not in (None, ""):
            out.append(float(v))
    return out


def summarize(rows: list[dict], duration_s: float | None = None) -> dict:
    """측정 구간(phase=measure) 요청만으로 요약한다."""
    measured = [r for r in rows if r.get("phase") == "measure"]
    ok = [r for r in measured if str(r.get("status")) == "200"]
    workload = measured[0]["workload"] if measured else (rows[0]["workload"] if rows else "")
    if duration_s is None and measured:
        sends = [float(r["t_send_unix"]) for r in measured]
        duration_s = max(sends) - min(sends) or None

    by_status: dict[str, int] = defaultdict(int)
    for r in measured:
        by_status[str(r.get("status"))] += 1

    s: dict = {
        "workload": workload,
        "requests": len(measured),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / len(measured), 4) if measured else None,
        "status_counts": dict(sorted(by_status.items())),
        "throughput_rps": round(len(ok) / duration_s, 3) if duration_s else None,
    }

    def pct(key: str, prefix: str) -> None:
        vals = _floats(ok, key)
        for q in (50, 95, 99):
            v = percentile(vals, q)
            s[f"{prefix}_p{q}_ms"] = round(v, 1) if v is not None else None
        s[f"{prefix}_mean_ms"] = round(sum(vals) / len(vals), 1) if vals else None

    pct("latency_ms", "latency")
    if workload == "llm":
        pct("client_ttft_ms", "ttft")
        pct("gw_queue_wait_ms", "gw_queue_wait")
        pct("gw_tpot_ms", "tpot")
        tokens = sum(_floats(ok, "output_tokens"))
        s["output_tokens_per_s"] = round(tokens / duration_s, 1) if duration_s else None
    elif workload == "cv":
        pct("cv_queue_wait_ms", "cv_queue_wait")
        pct("cv_batch_wait_ms", "cv_batch_wait")
        pct("cv_inference_ms", "cv_inference")
        sizes = _floats(ok, "cv_batch_size")
        s["cv_batch_size_mean"] = round(sum(sizes) / len(sizes), 2) if sizes else None
    return s


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


KEY_ORDER = [
    "workload", "requests", "ok", "error_rate", "throughput_rps",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "gw_queue_wait_p95_ms", "tpot_p50_ms", "tpot_p95_ms",
    "output_tokens_per_s",
    "cv_queue_wait_p50_ms", "cv_queue_wait_p95_ms", "cv_batch_wait_p95_ms", "cv_inference_p50_ms",
    "cv_inference_p95_ms", "cv_batch_size_mean",
]


def format_table(named: list[tuple[str, dict]]) -> str:
    """여러 실행의 요약을 세로 표로 (행 = 지표, 열 = 실행)."""
    keys = [k for k in KEY_ORDER if any(s.get(k) is not None for _, s in named)]
    names = [n for n, _ in named]
    w0 = max(len(k) for k in keys) + 2
    widths = [max(len(n), 10) + 2 for n in names]
    lines = ["".ljust(w0) + "".join(n.rjust(w) for n, w in zip(names, widths))]
    for k in keys:
        cells = []
        for (_, s), w in zip(named, widths):
            v = s.get(k)
            cells.append(("-" if v is None else str(v)).rjust(w))
        lines.append(k.ljust(w0) + "".join(cells))
    return "\n".join(lines)
