"""사용법

  # LLM에 초당 2건씩 60초 (앞 10초는 워밍업이라 요약에서 제외)
  python -m loadgen run --workload llm --rate 2 --duration 60

  # CV에 동시 사용자 8명
  python -m loadgen run --workload cv --concurrency 8 --duration 60

  # 여러 실행 비교
  python -m loadgen summary results/run-a results/run-b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from loadgen.runner import run
from loadgen.stats import format_table, read_csv, summarize


def _cmd_summary(args: argparse.Namespace) -> int:
    named = []
    for d in args.runs:
        path = Path(d)
        csv_path = path / "requests.csv" if path.is_dir() else path
        meta_path = csv_path.parent / "meta.json"
        duration = None
        if meta_path.is_file():
            duration = float(json.loads(meta_path.read_text())["args"]["duration"])
        named.append((csv_path.parent.name, summarize(read_csv(csv_path), duration_s=duration)))
    print(format_table(named))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m loadgen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="부하를 걸고 결과를 저장")
    r.add_argument("--workload", choices=["llm", "cv"], required=True)
    how = r.add_mutually_exclusive_group(required=True)
    how.add_argument("--rate", type=float, help="고정 도착률: 초당 요청 수 (open loop)")
    how.add_argument("--concurrency", type=int, help="고정 동시성: 동시에 요청하는 사용자 수 (closed loop)")
    r.add_argument("--arrival", choices=["poisson", "constant"], default="poisson",
                   help="--rate 일 때 요청 간격: poisson(무작위, 실제 트래픽에 가까움) / constant(일정)")
    r.add_argument("--duration", type=float, default=60, help="측정 구간 길이(초)")
    r.add_argument("--warmup", type=float, default=10, help="워밍업 길이(초). 이 구간은 요약에서 제외")
    r.add_argument("--target", default=os.getenv("GATEWAY_URL", "http://localhost:8000"))
    r.add_argument("--timeout", type=float, default=120, help="요청 1건의 최대 대기(초)")
    r.add_argument("--max-inflight", type=int, default=1000, help="부하 생성기의 동시 요청 한도")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--name", help="실행 이름 (결과 폴더 이름). 기본: 날짜-시각-워크로드")
    r.add_argument("--out", help="결과 폴더. 기본: results/<name>")
    r.add_argument("--env-file", default=".env", help="결과에 함께 남길 설정 파일")
    llm = r.add_argument_group("LLM")
    llm.add_argument("--input-words", type=int, default=64, help="프롬프트 본문 단어 수")
    llm.add_argument("--max-tokens", type=int, default=128, help="출력 토큰 수 상한")
    llm.add_argument("--fixed-output", action=argparse.BooleanOptionalAction, default=True,
                     help="출력 길이를 max-tokens로 고정 (ignore_eos)")
    cv = r.add_argument_group("CV")
    cv.add_argument("--images", default="data/val2017", help="이미지 폴더")
    cv.add_argument("--num-images", type=int, default=200, help="메모리에 올려 돌려 쓸 이미지 수")
    r.set_defaults(func=lambda a: asyncio.run(run(a)))

    s = sub.add_parser("summary", help="저장된 실행 결과를 표로 비교")
    s.add_argument("runs", nargs="+", help="결과 폴더 또는 requests.csv")
    s.set_defaults(func=_cmd_summary)

    args = p.parse_args(argv)
    if args.cmd == "run" and (args.rate is not None and args.rate <= 0 or
                              args.concurrency is not None and args.concurrency <= 0):
        p.error("--rate / --concurrency 는 0보다 커야 합니다")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
