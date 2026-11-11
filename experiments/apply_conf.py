#!/usr/bin/env python3
"""실험 설정 적용: .env 의 실험용 값을 기준(base)으로 되돌린 뒤 experiments/conf/<이름>.env 값으로 덮어쓴다.

  python3 experiments/apply_conf.py e2a-batch16        # .env 수정 (원본은 .env.bak)
  python3 experiments/apply_conf.py --list             # 쓸 수 있는 설정 목록
  python3 experiments/apply_conf.py --wait             # 재시작 후 게이트웨이·LLM·CV 워커가 준비될 때까지 대기

보통은 `make use CONF=e2a-batch16` 으로 부른다 (설정 적용 → docker compose up → 준비 대기).
실험마다 .env 를 손으로 고치면 되돌리는 걸 잊기 쉬워서, 실험 조건을 파일로 남기고 매번 base 에서 출발한다.
적용한 설정 이름은 EXP_CONF 로 .env 에 남고, 실험 실행기가 .env 를 결과 폴더(env.txt)에 복사하므로 결과에도 남는다.
표준 라이브러리만 사용.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT / "experiments" / "conf"
LINE = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)=(?P<value>[^#\n]*?)(?P<comment>\s+#.*)?$")


def parse(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        mt = LINE.match(ln.strip())
        if mt:
            out[mt["key"]] = mt["value"].strip()
    return out


def apply(env_path: Path, values: dict[str, str]) -> list[tuple[str, str | None, str]]:
    lines = env_path.read_text().splitlines() if env_path.is_file() else []
    changes, seen = [], set()
    for i, ln in enumerate(lines):
        mt = LINE.match(ln)
        if not mt or mt["key"] not in values:
            continue
        key, old, new = mt["key"], mt["value"].strip(), values[mt["key"]]
        seen.add(key)
        if old != new:
            # 값만 바꾸고 줄 끝 설명(# ...)은 같은 칸에 그대로 둔다
            comment = (mt["comment"] or "").lstrip()
            if comment:
                col = len(ln.rstrip()) - len(comment)
                lines[i] = f"{key}={new}".ljust(col - 1) + " " + comment
            else:
                lines[i] = f"{key}={new}"
            changes.append((key, old, new))
    missing = [k for k in values if k not in seen]
    if missing:
        lines += ["", "# ---- experiments/apply_conf.py 가 추가한 값"]
        for k in missing:
            lines.append(f"{k}={values[k]}")
            changes.append((k, None, values[k]))
    env_path.write_text("\n".join(lines) + "\n")
    return changes


def wait_ready(gateway: str, timeout: float, settle: float = 20.0) -> int:
    """게이트웨이 /readyz(Redis·LLM)와 /metrics 의 cv_workers_alive 가 기대한 워커 수에 닿을 때까지 기다린다.

    방금 교체된 워커의 heartbeat 키는 TTL(기본 15초) 동안 Redis에 남아 '살아 있음'으로 세진다.
    그래서 먼저 settle 초를 기다려 옛 키가 사라진 뒤에 센다 (새 워커가 뜨기 전에 실험이 시작되는 것을 막음).
    """
    env = parse(ROOT / ".env") if (ROOT / ".env").is_file() else {}
    want = int(env.get("CV_WORKERS", "1") or 1)
    if settle > 0:
        print(f"  옛 워커의 heartbeat가 사라지도록 {settle:.0f}초 기다립니다", flush=True)
        time.sleep(settle)
    t0, last = time.time(), 0.0
    state = "시작 전"
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{gateway}/readyz", timeout=3) as r:
                ready = b'"ready":true' in r.read().replace(b" ", b"")
        except Exception:
            ready = False
        alive = 0
        try:
            with urllib.request.urlopen(f"{gateway}/metrics", timeout=3) as r:
                for ln in r.read().decode().splitlines():
                    if ln.startswith("cv_workers_alive "):
                        alive = int(float(ln.split()[1]))
        except Exception:
            pass
        state = f"게이트웨이 준비={'예' if ready else '아니오'}, CV 워커 {alive}/{want}"
        if ready and alive >= want:
            print(f"[준비 완료] {state} ({time.time() - t0:.0f}초)")
            return 0
        if time.time() - last > 10:
            print(f"  기다리는 중 ({time.time() - t0:.0f}초): {state}  (vLLM 재시작은 1~3분 걸림)", flush=True)
            last = time.time()
        time.sleep(2)
    print(f"[시간 초과] {timeout:.0f}초 안에 준비되지 않았습니다: {state}\n  make ps, make logs 로 확인하세요.")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("conf", nargs="?", help="experiments/conf/<이름>.env 의 이름 (예: e2a-batch16, base)")
    p.add_argument("--env", default=str(ROOT / ".env"), help="고칠 .env 경로")
    p.add_argument("--list", action="store_true", help="쓸 수 있는 설정 목록")
    p.add_argument("--wait", action="store_true", help="설정 대신, 서비스가 준비될 때까지 기다리기만 한다")
    p.add_argument("--gateway", default="http://localhost:8000")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--settle", type=float, default=20, help="--wait 전에 기다릴 초 (옛 heartbeat 만료용, CV_HEARTBEAT_TTL_S + 5)")
    args = p.parse_args()

    if args.wait:
        return wait_ready(args.gateway, args.timeout, args.settle)
    if args.list or not args.conf:
        for f in sorted(CONF_DIR.glob("*.env")):
            first = next((ln[1:].strip() for ln in f.read_text().splitlines() if ln.startswith("#")), "")
            print(f"  {f.stem:<16} {first}")
        return 0 if args.list else 1

    base, conf = CONF_DIR / "base.env", CONF_DIR / f"{args.conf}.env"
    if not conf.is_file():
        print(f"없는 설정입니다: {conf.relative_to(ROOT)}\n쓸 수 있는 설정:")
        for f in sorted(CONF_DIR.glob("*.env")):
            print(f"  {f.stem}")
        return 1
    values = {**parse(base), **parse(conf), "EXP_CONF": args.conf}
    env_path = Path(args.env)
    if env_path.is_file():
        shutil.copy(env_path, env_path.with_name(env_path.name + ".bak"))
    changes = apply(env_path, values)
    print(f"[{args.conf}] {conf.relative_to(ROOT)} 적용 → {env_path.name}")
    if not changes:
        print("  바뀐 값 없음 (이미 이 설정)")
    for key, old, new in changes:
        print(f"  {key}: {old if old is not None else '(없음)'} → {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
