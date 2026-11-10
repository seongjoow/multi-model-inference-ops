#!/usr/bin/env python3
"""실험 실행기: 단독·경합 시나리오를 단계별로 돌리고, 단계마다 자원 사용량까지 저장한다.

  python3 experiments/run.py --name d3-pilot --repeat 1 --duration 20 --warmup 5     # 시험 실행 (약 10분)
  python3 experiments/run.py --name d3 --repeat 3                                     # 본 실행

시나리오 (--scenarios 로 고름)
  llm-only        LLM만, --llm-rates 의 각 도착률
  cv-only         CV만, --cv-rates 의 각 도착률
  contention-cv-sweep   LLM을 --llm-fixed 로 고정하고 CV를 --cv-rates 로 올림 → CV가 LLM에 주는 영향
  contention-llm-sweep  CV를 --cv-fixed 로 고정하고 LLM을 --llm-rates 로 올림 → LLM이 CV에 주는 영향
  (예전 이름 mixed-cv-sweep, mixed-llm-sweep 도 받아서 새 이름으로 바꿔 실행한다)

결과: results/<name>/
  plan.json, env.txt            실행 조건 (옵션, 커밋, GPU, 당시 .env)
  runs.csv                      단계 목록 (시나리오, 반복, 도착률, 측정 구간, 결과 폴더)
  <시나리오>/rep<k>/llm<r>_cv<r>/
      llm/, cv/                 부하 생성기 결과 (requests.csv, summary.json, meta.json)
      resources.json            측정 구간의 GPU·CPU·큐 상태 (Prometheus에서 조회)

의존성 없이 표준 라이브러리만 쓴다 (EC2 호스트의 python3로 바로 실행).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ["llm-only", "cv-only", "contention-cv-sweep", "contention-llm-sweep"]
# d3까지 쓰던 이름. 분석(report.py)도 같은 표로 옛 결과를 새 이름으로 읽는다
SCENARIO_ALIASES = {"mixed-cv-sweep": "contention-cv-sweep", "mixed-llm-sweep": "contention-llm-sweep"}

# 측정 구간 동안 Prometheus에서 평균·최대를 뽑을 지표
RESOURCE_QUERIES = {
    "gpu_util_pct": "max(nvidia_gpu_utilization_percent)",
    "gpu_mem_used_gb": "sum(nvidia_gpu_memory_used_bytes) / 1e9",
    "gpu_power_w": "sum(nvidia_gpu_power_watts)",
    "host_cpu_pct": '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[10s])))',
    "cpu_cores_by_job": "sum by (job) (rate(process_cpu_seconds_total[10s]))",
    "llm_server_running": "sum(vllm:num_requests_running) or sum(llamacpp:requests_processing)",
    "llm_server_waiting": "sum(vllm:num_requests_waiting) or sum(llamacpp:requests_deferred)",
    "llm_kv_cache_pct": "100 * (max(vllm:kv_cache_usage_perc) or max(vllm:gpu_cache_usage_perc))",
    "gateway_llm_waiting": "max(gateway_llm_waiting)",
    "cv_queue_depth": "max(cv_queue_depth)",
    "cv_batch_size": "sum(rate(cv_worker_batch_size_sum[10s])) / sum(rate(cv_worker_batch_size_count[10s]))",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_rates(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def fmt_rate(r: float | None) -> str:
    if r is None:
        return "0"
    return str(int(r)) if float(r).is_integer() else str(r).replace(".", "p")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def compose_cmd(args: argparse.Namespace) -> list[str]:
    if args.compose:
        return shlex.split(args.compose)
    cmd = ["docker", "compose"]
    if shutil.which("nvidia-smi"):  # Makefile과 같은 규칙: GPU가 있으면 EC2 구성
        cmd += ["-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml"]
    return cmd


def loadgen_cmd(args: argparse.Namespace, run_args: list[str]) -> list[str]:
    if args.launcher == "local":
        return [sys.executable, "-m", "loadgen", *run_args]
    return [*compose_cmd(args), "--profile", "tools", "run", "--rm", "--no-deps", "-T", "loadgen", *run_args]


def http_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def build_steps(args: argparse.Namespace) -> list[dict]:
    steps = []
    for rep in range(1, args.repeat + 1):
        for sc in args.scenarios:
            if sc == "llm-only":
                pairs = [(r, None) for r in args.llm_rates]
            elif sc == "cv-only":
                pairs = [(None, r) for r in args.cv_rates]
            elif sc == "contention-cv-sweep":
                pairs = [(args.llm_fixed, r) for r in args.cv_rates]
            else:
                pairs = [(r, args.cv_fixed) for r in args.llm_rates]
            for llm, cv in pairs:
                steps.append({"rep": rep, "scenario": sc, "llm_rate": llm, "cv_rate": cv})
    return steps


def snapshot_resources(prom: str, start: float, end: float) -> dict:
    """측정 구간 [start, end]의 각 지표를 2초 간격으로 읽어 평균·최대를 낸다."""
    out: dict = {"window": {"start_unix": start, "end_unix": end}}
    for name, expr in RESOURCE_QUERIES.items():
        q = urllib.parse.urlencode({"query": expr, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": "2"})
        try:
            data = http_json(f"{prom}/api/v1/query_range?{q}")["data"]["result"]
        except Exception as exc:  # Prometheus가 없거나 지표가 없으면 비워 둔다
            out[name] = {"error": repr(exc)}
            continue

        def stats(values: list) -> dict:
            xs = [float(v) for _, v in values if v not in ("NaN", "+Inf", "-Inf")]
            return {"mean": round(sum(xs) / len(xs), 4), "max": round(max(xs), 4)} if xs else {"mean": None, "max": None}

        if name.endswith("_by_job"):
            out[name] = {s["metric"].get("job", "?"): stats(s["values"]) for s in data}
        else:
            out[name] = stats(data[0]["values"]) if data else {"mean": None, "max": None}
    return out


def run_step(args: argparse.Namespace, exp_dir: Path, step: dict, env: dict) -> dict:
    llm, cv = step["llm_rate"], step["cv_rate"]
    step_dir = exp_dir / step["scenario"] / f"rep{step['rep']}" / f"llm{fmt_rate(llm)}_cv{fmt_rate(cv)}"
    step_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed + step["rep"] - 1  # 반복마다 다른 도착 간격, 같은 반복 번호면 다시 실행해도 같음

    procs = []
    for workload, rate, extra in (("llm", llm, args.llm_extra), ("cv", cv, args.cv_extra)):
        if rate is None:
            continue
        out_rel = (step_dir / workload).relative_to(ROOT)
        run_args = ["run", "--workload", workload, "--rate", str(rate), "--duration", str(args.duration),
                    "--warmup", str(args.warmup), "--seed", str(seed), "--out", str(out_rel),
                    "--name", f"{exp_dir.name}-{step['scenario']}-r{step['rep']}-{workload}{fmt_rate(rate)}",
                    *shlex.split(extra)]
        logf = open(step_dir / f"loadgen-{workload}.log", "w")
        procs.append((workload, subprocess.Popen(loadgen_cmd(args, run_args), cwd=ROOT, env=env,
                                                 stdout=logf, stderr=subprocess.STDOUT), logf))

    codes = {}
    for workload, p, logf in procs:
        codes[workload] = p.wait()
        logf.close()

    row = {**step, "step_dir": str(step_dir.relative_to(exp_dir)), "status": "ok"}
    starts, ends = [], []
    for workload, code in codes.items():
        meta_path = step_dir / workload / "meta.json"
        if code != 0 or not meta_path.is_file():
            row["status"] = f"failed:{workload}"
            continue
        meta = json.loads(meta_path.read_text())
        starts.append(meta["measure_start_unix"])
        ends.append(meta["measure_start_unix"] + args.duration)
    if starts:
        start, end = min(starts), max(ends)
        row.update(measure_start_unix=round(start, 3), measure_end_unix=round(end, 3))
        res = snapshot_resources(args.prometheus, start, end)
        (step_dir / "resources.json").write_text(json.dumps(res, indent=2))
        row["grafana"] = f"{args.grafana}/d/inference-ops/?from={int(start * 1000)}&to={int(end * 1000)}"
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=f"exp-{datetime.now().strftime('%Y%m%d-%H%M')}", help="결과 폴더 이름")
    p.add_argument("--scenarios", default=",".join(SCENARIOS), help=f"쉼표로 구분: {','.join(SCENARIOS)}")
    p.add_argument("--llm-rates", default="1,2,4,8,12", help="LLM 도착률 단계 (req/s)")
    p.add_argument("--cv-rates", default="10,25,50,100", help="CV 도착률 단계 (req/s)")
    p.add_argument("--llm-fixed", type=float, default=4, help="contention-cv-sweep 에서 고정할 LLM 도착률")
    p.add_argument("--cv-fixed", type=float, default=25, help="contention-llm-sweep 에서 고정할 CV 도착률")
    p.add_argument("--repeat", type=int, default=3, help="전체 반복 횟수")
    p.add_argument("--duration", type=float, default=60, help="단계별 측정 구간(초)")
    p.add_argument("--warmup", type=float, default=10, help="단계별 워밍업(초)")
    p.add_argument("--cooldown", type=float, default=10, help="단계 사이 쉬는 시간(초)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-extra", default="", help='LLM 부하 생성기에 넘길 추가 옵션 (예: "--max-tokens 64")')
    p.add_argument("--cv-extra", default="", help='CV 부하 생성기에 넘길 추가 옵션 (예: "--num-images 300")')
    p.add_argument("--launcher", choices=["docker", "local"], default="docker",
                   help="docker: loadgen 컨테이너로 실행 / local: 현재 파이썬으로 직접 실행")
    p.add_argument("--compose", help="docker compose 명령을 직접 지정 (기본: GPU 유무로 자동)")
    p.add_argument("--gateway", default="http://localhost:8000")
    p.add_argument("--prometheus", default="http://localhost:9090")
    p.add_argument("--grafana", default="http://localhost:3000")
    p.add_argument("--yes", action="store_true", help="시작 전 확인 없이 바로 실행")
    args = p.parse_args()
    args.scenarios = list(dict.fromkeys(SCENARIO_ALIASES.get(s, s) for s in args.scenarios.split(",") if s))
    bad = [s for s in args.scenarios if s not in SCENARIOS]
    if bad:
        p.error(f"모르는 시나리오: {bad}")
    args.llm_rates, args.cv_rates = parse_rates(args.llm_rates), parse_rates(args.cv_rates)

    exp_dir = ROOT / "results" / args.name
    if exp_dir.exists():
        p.error(f"이미 있는 실험 이름입니다: {exp_dir}")

    # 시작 전 점검: 서비스가 준비됐는지, Prometheus가 보이는지
    try:
        ready = http_json(f"{args.gateway}/readyz")
    except Exception as exc:
        ready = {"ready": False, "error": repr(exc)}
    if not ready.get("ready"):
        print(f"게이트웨이가 준비되지 않았습니다: {ready}\nmake ps / make smoke 로 먼저 확인하세요.")
        return 1
    try:
        targets = http_json(f"{args.prometheus}/api/v1/targets")["data"]["activeTargets"]
        down = [t["labels"]["job"] for t in targets if t["health"] != "up"]
        if down:
            log(f"[경고] Prometheus 수집 대상 중 DOWN: {down} (자원 기록이 비게 됩니다)")
    except Exception as exc:
        log(f"[경고] Prometheus에 연결할 수 없습니다 ({exc}). 자원 기록 없이 진행합니다.")

    steps = build_steps(args)
    per_step = args.warmup + args.duration + args.cooldown + 10
    log(f"실험 '{args.name}': {len(steps)}단계 × 약 {per_step:.0f}초 ≈ {len(steps) * per_step / 60:.0f}분 예상")
    for sc in args.scenarios:
        n = sum(1 for s in steps if s["scenario"] == sc and s["rep"] == 1)
        log(f"  {sc}: {n}단계 × {args.repeat}회")
    if not args.yes and sys.stdin.isatty():
        if input("시작할까요? [y/N] ").strip().lower() != "y":
            return 1

    exp_dir.mkdir(parents=True)
    commit = git_commit()
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else "none"
    plan = {
        "name": args.name, "started_at": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
        "gpu": gpu, "args": {k: v for k, v in vars(args).items()}, "resource_queries": RESOURCE_QUERIES,
    }
    (exp_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    if (ROOT / ".env").is_file():
        shutil.copy(ROOT / ".env", exp_dir / "env.txt")
    if commit.endswith("-dirty"):
        log("[경고] 커밋하지 않은 변경이 있습니다. 결과가 정확히 어떤 코드였는지 남기려면 먼저 커밋하세요.")

    env = {**os.environ, "GIT_COMMIT": commit}
    env.setdefault("GATEWAY_URL", args.gateway)  # local 실행용 (docker는 compose 설정을 씀)
    if args.launcher == "docker":
        # 경합 단계는 부하 생성기 두 개를 동시에 띄우므로, 이미지를 미리 만들어 동시 빌드 충돌을 막는다
        log("부하 생성기 이미지 준비")
        subprocess.run([*compose_cmd(args), "--profile", "tools", "build", "loadgen"], cwd=ROOT, env=env, check=True,
                       stdout=subprocess.DEVNULL)
    fields = ["rep", "scenario", "llm_rate", "cv_rate", "status", "step_dir",
              "measure_start_unix", "measure_end_unix", "grafana"]
    index = open(exp_dir / "runs.csv", "w", newline="")
    writer = csv.DictWriter(index, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()

    interrupted = False

    def _stop(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _stop)
    t0 = time.time()
    try:
        for i, step in enumerate(steps, 1):
            if interrupted:
                break
            label = f"{step['scenario']} rep{step['rep']} llm={step['llm_rate']} cv={step['cv_rate']}"
            log(f"({i}/{len(steps)}) {label}")
            row = run_step(args, exp_dir, step, env)
            writer.writerow(row)
            index.flush()
            if row["status"] != "ok":
                log(f"  ! {row['status']} — {exp_dir / row['step_dir']} 의 loadgen-*.log 확인")
            if i < len(steps) and not interrupted:
                time.sleep(args.cooldown)
    finally:
        index.close()

    log(f"끝 ({(time.time() - t0) / 60:.1f}분){' — 중간에 멈춤' if interrupted else ''}")
    print(f"\n결과: {exp_dir}\n분석: python analysis/report.py {exp_dir.relative_to(ROOT)}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
