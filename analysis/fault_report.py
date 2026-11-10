#!/usr/bin/env python3
"""장애 재현(E5) 결과 분석: 언제 죽었고, 무엇이 그걸 알아챘고, 무엇을 잃었고, 언제 회복됐는가.

  python analysis/fault_report.py results/e5-simple results/e5-reliable
  make fault-report EXPS="e5-simple e5-reliable"

만드는 것 (기본: 실험 하나면 results/<이름>/report/, 여럿이면 results/compare/fault__<이름들>/)
  fault_report.md   실험별 비교표 (유실 의심 요청, 되살린 작업, 적체, 회복 시간, 경보가 울린 시점)
  fig_fault_<이름>.png   시간축 그림: 응답 상태, 요청별 지연, 큐 길이, 살아 있는 워커 수 (장애 구간 음영)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as rp  # noqa: E402


def series(prom: dict, name: str, t0: float) -> list[tuple[dict, pd.Series]]:
    out = []
    for s in prom.get("results", {}).get(name, []):
        if "values" not in s:
            continue
        idx = [float(t) - t0 for t, _ in s["values"]]
        vals = [float(v) if v not in ("NaN", "+Inf", "-Inf") else float("nan") for _, v in s["values"]]
        out.append((s.get("metric", {}), pd.Series(vals, index=idx)))
    return out


def load_run(d: Path) -> dict:
    tl = pd.read_csv(d / "timeline.csv")
    t0 = float(tl.loc[tl["event"] == "load_start", "t_unix"].iloc[0])
    ev = {e: float(t) - t0 for e, t in zip(tl["event"], tl["t_unix"])}
    detail = tl.loc[tl["event"] == "load_start", "detail"].iloc[0]
    info = dict(kv.split("=", 1) for kv in str(detail).split() if "=" in kv)
    cv = pd.read_csv(d / "cv" / "requests.csv")
    cv["t_send"] = pd.to_numeric(cv["t_send_unix"], errors="coerce") - t0
    cv["t_done"] = cv["t_send"] + pd.to_numeric(cv["latency_ms"], errors="coerce") / 1000
    cv["status"] = cv["status"].astype(str)
    prom = json.loads((d / "prometheus.json").read_text()) if (d / "prometheus.json").is_file() else {}
    worker_log = (d / "logs" / "cv-worker.log").read_text() if (d / "logs" / "cv-worker.log").is_file() else ""
    recovered = 0
    for ln in worker_log.splitlines():
        if '"recovered_jobs"' in ln and "{" in ln:
            try:
                recovered += int(json.loads(ln[ln.index("{"):]).get("count", 0))
            except ValueError:
                pass
    return {"name": d.name, "dir": d, "t0": t0, "ev": ev, "info": info, "cv": cv, "prom": prom, "recovered": recovered}


def first_after(s: pd.Series, t: float, cond) -> float | None:
    hit = s[(s.index >= t) & cond(s)]
    return float(hit.index[0]) if len(hit) else None


def summarize(run: dict) -> list[str]:
    ev, cv = run["ev"], run["cv"]
    kill = ev.get("worker_kill")
    up = ev.get("worker_start", ev.get("worker_auto_restarted"))
    ok = cv[cv["status"] == "200"]
    before = ok[ok["t_send"] < kill] if kill is not None else ok
    after = ok[(ok["t_send"] >= (up or 0)) & (ok["t_send"] < (up or 0) + 60)] if up is not None else ok.iloc[0:0]
    lat = lambda df: rp.fmt(rp.q(df["latency_ms"], 95))  # noqa: E731
    queue = series(run["prom"], "cv_queue_depth", run["t0"])
    qmax, cleared = "-", "-"
    if queue:
        q = queue[0][1]
        if kill is not None:
            qmax = rp.fmt(q[q.index >= kill].max())
        if up is not None:
            t = first_after(q, up + 1, lambda s: s <= 5)
            cleared = f"{t - up:.0f}" if t is not None else "끝까지 안 됨"
    alerts = []
    for metric, s in series(run["prom"], "alerts", run["t0"]):
        s = s.dropna()
        if len(s) and kill is not None:
            first = float(s.index[0])
            alerts.append(f"{metric.get('alertname', '?')} (+{first - kill:.0f}s)")
    lost = int((cv["status"] == "504").sum())
    other = cv[~cv["status"].isin(["200", "504"])]["status"].value_counts()
    return [run["name"], run["info"].get("queue_mode", "?"), run["info"].get("mode", "?"),
            run["info"].get("workers", "?"), str(lost), str(run["recovered"]),
            ", ".join(f"{k}×{v}" for k, v in other.items()) or "-", qmax,
            f"{up - kill:.0f}" if kill is not None and up is not None else "-", cleared, lat(before), lat(after),
            "; ".join(sorted(alerts)) or "없음"]


def fig_run(run: dict, out: Path) -> Path:
    ev, cv = run["ev"], run["cv"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.4, 1, 0.6]})
    kill = ev.get("worker_kill")
    up = ev.get("worker_start", ev.get("worker_auto_restarted"))
    for ax in axes:
        if kill is not None and up is not None:
            ax.axvspan(kill, up, color=rp.GRID, alpha=0.9, lw=0)
        for t, label in ((kill, "워커 종료"), (up, "워커 재시작")):
            if t is not None:
                ax.axvline(t, color=rp.REF, linestyle="--", linewidth=1)
    # (1) 완료 시각 기준 초당 응답 수
    done = cv.dropna(subset=["t_done"]).copy()
    done["sec"] = done["t_done"].astype(int)
    per = done.groupby(["sec", "status"]).size().unstack(fill_value=0)
    if len(per):  # 응답이 하나도 없던 초도 0으로 그려야 장애 구간이 보인다
        per = per.reindex(range(int(per.index.min()), int(per.index.max()) + 1), fill_value=0)
    if "200" in per:
        axes[0].plot(per.index, per["200"], color=rp.C1, label="200 성공")
    for i, st in enumerate([c for c in per.columns if c != "200"]):
        axes[0].bar(per.index, per[st], color=[rp.C2, rp.C3, "#7b5cd6"][i % 3], width=1.0, label=st)
    axes[0].set(title="CV 응답 수 (끝난 시각 기준, 초당)", ylabel="건/초")
    axes[0].legend(loc="upper right", ncol=4)
    # (2) 요청별 지연 (보낸 시각 기준)
    okm = cv["status"] == "200"
    axes[1].scatter(cv.loc[okm, "t_send"], cv.loc[okm, "latency_ms"], s=4, color=rp.C1, label="200", alpha=0.6)
    axes[1].scatter(cv.loc[~okm, "t_send"], cv.loc[~okm, "latency_ms"], s=10, color=rp.C2, label="실패", alpha=0.9)
    axes[1].axhline(rp.SLO["cv_ms"], color=rp.REF, linewidth=1)
    axes[1].annotate(f"SLO {rp.SLO['cv_ms']:g} ms", xy=(0.995, rp.SLO["cv_ms"]), xycoords=("axes fraction", "data"),
                     xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=rp.INK2)
    axes[1].set_yscale("log")
    axes[1].set(title="CV 요청별 전체 지연 (보낸 시각 기준)", ylabel="ms")
    axes[1].legend(loc="upper right")
    # (3) 큐 길이, (4) 살아 있는 워커
    for _, s in series(run["prom"], "cv_queue_depth", run["t0"]):
        axes[2].plot(s.index, s.values, color=rp.C1)
    axes[2].set(title="CV 큐 길이 (Prometheus)", ylabel="작업 수")
    axes[2].set_ylim(bottom=0)
    for _, s in series(run["prom"], "cv_workers_alive", run["t0"]):
        axes[3].step(s.index, s.values, where="post", color=rp.C1)
    axes[3].set(title="살아 있는 워커 수 (heartbeat 기준)", ylabel="개", xlabel="부하 시작 후 시간 (초)")
    axes[3].set_ylim(bottom=0)
    alerts = series(run["prom"], "alerts", run["t0"])
    for i, (metric, s) in enumerate(alerts):
        s = s.dropna()
        if len(s):
            t = float(s.index[0])
            axes[2].annotate(f"경보: {metric.get('alertname', '?')}", xy=(t, 0), xycoords=("data", "axes fraction"),
                             xytext=(2, 6 + 12 * i), textcoords="offset points", fontsize=8, color=rp.INK)
    fig.suptitle(f"장애 재현: {run['name']} (큐 모드 {run['info'].get('queue_mode', '?')}, 음영 = 워커가 없던 구간)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=rp.INK)
    fig.tight_layout()
    path = out / f"fig_fault_{run['name']}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dirs", nargs="+")
    p.add_argument("--out")
    args = p.parse_args()
    dirs = [Path(x) for x in args.exp_dirs]
    if args.out:
        out = Path(args.out)
    elif len(dirs) == 1:
        out = dirs[0] / "report"
    else:
        out = Path("results") / "compare" / ("fault__" + "__".join(d.name for d in dirs))
    out.mkdir(parents=True, exist_ok=True)
    rp.setup_style()
    runs = [load_run(d) for d in dirs]
    figs = [fig_run(r, out) for r in runs]
    lines = ["# 장애 재현 결과", "",
             "- 유실 의심 = 504 (게이트웨이가 결과를 `CV_WAIT_TIMEOUT_S` 동안 기다렸지만 끝내 받지 못한 요청)",
             "- 되살린 작업 = 워커가 재시작할 때 처리 목록에서 큐로 되돌린 작업 수 (reliable 모드만)",
             "- 복구 = 종료부터 워커가 다시 뜰 때까지(초), 적체 해소 = 재시작 후 큐 길이가 5 이하로 돌아올 때까지(초)",
             "- 경보 옆 +N s = 워커 종료 후 처음 firing 이 된 시점", "",
             rp.md_table(["실험", "큐 모드", "장애 방식", "워커 수", "504 (유실 의심)", "되살린 작업", "그 밖의 실패",
                          "장애 중 최대 큐", "복구 (초)", "적체 해소 (초)", "장애 전 CV p95 (ms)",
                          "재시작 후 60초 CV p95 (ms)", "울린 경보"], [summarize(r) for r in runs]), ""]
    lines += [f"![{f.stem}]({f.name})" for f in figs] + [""]
    (out / "fault_report.md").write_text("\n".join(lines))
    print(f"장애 실험 {len(runs)}개 → {out}/")
    for f in [out / "fault_report.md", *figs]:
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
