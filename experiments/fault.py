#!/usr/bin/env python3
"""E5 장애 재현: LLM·CV 부하를 거는 동안 CV 워커 하나를 강제로 죽이고, 탐지·복구 과정을 기록한다.

  make fault-worker NAME=e5-simple                       # 기본: 60초에 죽이고 30초 뒤 수동 재시작
  make fault-worker NAME=e5-crash MODE=crash             # 프로세스만 죽여 Docker가 자동 재시작하게
  python3 experiments/fault.py --name e5-simple --kill-at 60 --down 30 --cv-rate 20 --llm-rate 4 --duration 180

장애 방식 (--mode)
  down  : docker kill (SIGKILL). Docker는 사람이 멈춘 것으로 보고 자동 재시작하지 않는다.
          --down 초 뒤에 docker start 로 수동 복구 → "워커 0개" 경보, 큐 적체, 복구 후 적체 해소를 볼 수 있다.
  crash : 컨테이너 안 프로세스를 호스트에서 kill -9 (메모리 부족 등으로 죽는 상황 흉내).
          Docker 재시작 정책(restart: unless-stopped)이 바로 다시 띄운다 → 자동 복구, 짧은 장애. sudo 필요.

결과 results/<name>/
  cv/, llm/         부하 생성기 요청별 기록 (requests.csv)
  timeline.csv      부하 시작·워커 종료·재시작·부하 끝 시각
  prometheus.json   장애 전후의 큐 길이, 살아 있는 워커 수, 워커 처리 속도, 응답 상태별 속도, 경보(ALERTS)
  logs/             그 구간의 cv-worker, gateway 로그
  env.txt           당시 .env
  summary.txt       요약 (상태별 건수, 유실 의심 요청, 되살린 작업, 울린 경보)

분석은 Mac에서: python analysis/fault_report.py results/e5-simple results/e5-reliable
표준 라이브러리만 사용.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROM_QUERIES = {
    "cv_queue_depth": "max(cv_queue_depth)",
    "cv_processing_jobs": "max(cv_processing_jobs)",
    "cv_workers_alive": "max(cv_workers_alive)",
    "cv_worker_done_rps": 'sum(rate(cv_worker_jobs_total{outcome="done"}[4s]))',
    "cv_status_rps": 'sum by (status) (rate(gateway_requests_total{workload="cv"}[4s]))',
    "llm_ttft_p95_s": "histogram_quantile(0.95, sum by (le) (rate(gateway_llm_ttft_seconds_bucket[10s])))",
    "alerts": 'ALERTS{alertstate="firing"}',
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def compose_cmd(args: argparse.Namespace) -> list[str]:
    if args.compose:
        return shlex.split(args.compose)
    cmd = ["docker", "compose"]
    if shutil.which("nvidia-smi"):  # Makefile과 같은 규칙: GPU가 있으면 EC2 구성
        cmd += ["-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml"]
    return cmd


def sh(cmd: list[str], check: bool = True) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=True, text=True).stdout.strip()


def prom_range(prom: str, expr: str, start: float, end: float) -> list:
    q = urllib.parse.urlencode({"query": expr, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": "1"})
    try:
        with urllib.request.urlopen(f"{prom}/api/v1/query_range?{q}", timeout=20) as r:
            return json.loads(r.read())["data"]["result"]
    except Exception as exc:  # Prometheus가 없어도 실험 기록은 남긴다
        return [{"error": repr(exc)}]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="결과 폴더 이름 (results/<name>)")
    p.add_argument("--mode", choices=["down", "crash"], default="down")
    p.add_argument("--kill-at", type=float, default=60, help="부하 시작 후 몇 초에 워커를 죽일지")
    p.add_argument("--down", type=float, default=30, help="mode=down: 죽인 뒤 몇 초 후에 다시 띄울지")
    p.add_argument("--duration", type=float, default=180, help="부하 길이(초)")
    p.add_argument("--cv-rate", type=float, default=20)
    p.add_argument("--llm-rate", type=float, default=4, help="0이면 LLM 부하 없이")
    p.add_argument("--compose", help="docker compose 명령을 직접 지정 (기본: GPU 유무로 자동)")
    p.add_argument("--prometheus", default="http://localhost:9090")
    args = p.parse_args()

    out = ROOT / "results" / args.name
    if out.exists() and any(out.iterdir()):
        print(f"결과 폴더가 이미 있습니다: {out.relative_to(ROOT)} (--name 을 바꾸세요)")
        return 1
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if (ROOT / ".env").is_file():
        shutil.copy(ROOT / ".env", out / "env.txt")
    compose = compose_cmd(args)
    workers = sh([*compose, "ps", "-q", "cv-worker"]).split()
    if not workers:
        print("실행 중인 cv-worker 가 없습니다. make up 으로 먼저 띄우세요.")
        return 1
    target = workers[0]
    hostname = sh(["docker", "inspect", "-f", "{{.Config.Hostname}}", target])
    queue_mode = next((ln.split("=", 1)[1].split("#")[0].strip() for ln in (out / "env.txt").read_text().splitlines()
                       if ln.startswith("CV_QUEUE_MODE=")), "?") if (out / "env.txt").is_file() else "?"

    timeline = open(out / "timeline.csv", "w", newline="")
    tw = csv.writer(timeline)
    tw.writerow(["t_unix", "event", "detail"])

    def mark(event: str, detail: str = "") -> float:
        t = time.time()
        tw.writerow([f"{t:.3f}", event, detail])
        timeline.flush()
        log(f"{event} {detail}")
        return t

    log(f"워커 {len(workers)}개 중 {target[:12]} (worker_id={hostname}) 를 {args.kill_at:g}초에 종료, 방식={args.mode}, "
        f"큐 모드={queue_mode}")
    log("부하 생성기 이미지 준비")
    subprocess.run([*compose, "--profile", "tools", "build", "loadgen"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    env = {**os.environ, "GIT_COMMIT": os.environ.get("GIT_COMMIT", "unknown")}
    procs = []
    t_start = mark("load_start", f"cv={args.cv_rate:g}/s llm={args.llm_rate:g}/s duration={args.duration:g}s "
                                 f"workers={len(workers)} queue_mode={queue_mode} mode={args.mode}")
    for workload, rate in (("cv", args.cv_rate), ("llm", args.llm_rate)):
        if rate <= 0:
            continue
        cmd = [*compose, "--profile", "tools", "run", "--rm", "--no-deps", "-T", "loadgen", "run",
               "--workload", workload, "--rate", str(rate), "--duration", str(args.duration), "--warmup", "0",
               "--out", f"results/{args.name}/{workload}"]
        f = open(out / "logs" / f"loadgen-{workload}.log", "w")
        procs.append((workload, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT), f))

    time.sleep(max(0.0, t_start + args.kill_at - time.time()))
    if args.mode == "down":
        mark("worker_kill", f"docker kill {target[:12]} (자동 재시작 안 됨)")
        sh(["docker", "kill", target])
        time.sleep(min(5.0, args.down))
        if sh(["docker", "inspect", "-f", "{{.State.Running}}", target], check=False) == "true":
            mark("worker_auto_restarted", "Docker가 자동으로 다시 띄움 (예상과 다름: 수동 복구 단계는 의미 없음)")
        time.sleep(max(0.0, args.down - 5.0))
        sh(["docker", "start", target])
        mark("worker_start", f"docker start {target[:12]} (수동 복구)")
    else:
        pid = sh(["docker", "inspect", "-f", "{{.State.Pid}}", target])
        restarts0 = sh(["docker", "inspect", "-f", "{{.RestartCount}}", target])
        mark("worker_kill", f"kill -9 host pid {pid} (Docker 재시작 정책이 다시 띄움)")
        kill = ["kill", "-9", pid] if os.geteuid() == 0 else ["sudo", "kill", "-9", pid]
        subprocess.run(kill, check=True)
        for _ in range(120):  # 최대 2분 동안 재시작 확인
            time.sleep(1)
            running = sh(["docker", "inspect", "-f", "{{.State.Running}} {{.RestartCount}}", target], check=False)
            if running.startswith("true") and running.split()[-1] != restarts0:
                mark("worker_start", f"자동 재시작 (RestartCount {restarts0} → {running.split()[-1]})")
                break
        else:
            mark("worker_start_timeout", "2분 안에 재시작되지 않음")

    for workload, proc, f in procs:
        proc.wait()
        f.close()
    t_end = mark("load_end")
    timeline.close()

    # 장애 전후 지표와 경보, 로그 저장
    log("Prometheus 지표·경보 저장")
    prom = {name: prom_range(args.prometheus, expr, t_start - 30, t_end + 5) for name, expr in PROM_QUERIES.items()}
    (out / "prometheus.json").write_text(json.dumps({"queries": PROM_QUERIES, "results": prom}, ensure_ascii=False))
    since = datetime.fromtimestamp(t_start - 5, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for svc in ("cv-worker", "gateway"):
        (out / "logs" / f"{svc}.log").write_text(
            sh([*compose, "logs", "--no-color", "--since", since, svc], check=False))

    # 요약
    lines = [f"장애 재현: {args.name}", f"  방식={args.mode}, 큐 모드={queue_mode}, 워커 {len(workers)}개 중 1개 종료",
             f"  부하: CV {args.cv_rate:g}/s, LLM {args.llm_rate:g}/s, {args.duration:g}초 (종료 시점 {args.kill_at:g}초)", ""]
    for workload, _, _ in procs:
        path = out / workload / "requests.csv"
        if not path.is_file():
            continue
        rows = list(csv.DictReader(open(path)))
        counts = Counter(r["status"] for r in rows)
        lines.append(f"  {workload.upper()} 응답: " + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))
        if workload == "cv":
            lost = [r for r in rows if r["status"] == "504"]
            lines.append(f"  CV 504 (결과를 못 받은 요청 = 유실 의심): {len(lost)}건")
    worker_log = (out / "logs" / "cv-worker.log").read_text()
    recovered = sum(int(json.loads(ln[ln.index("{"):]).get("count", 0)) for ln in worker_log.splitlines()
                    if '"recovered_jobs"' in ln and "{" in ln)
    lines.append(f"  재시작 때 되살린 작업(reliable): {recovered}건")
    fired = sorted({s["metric"].get("alertname", "?") for s in prom.get("alerts", []) if "metric" in s})
    lines.append(f"  울린 경보: {', '.join(fired) if fired else '없음'}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"끝. 결과: results/{args.name}/  (Mac에서 python analysis/fault_report.py results/{args.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
