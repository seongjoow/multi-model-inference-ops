#!/usr/bin/env python3
"""실험 결과 분석: results/<name>/ 을 읽어 표·그래프·요약 문서를 만든다.

  python analysis/report.py results/d3

만드는 것 (results/<name>/report/)
  runs.csv          단계×반복 1행: 지연 백분위, 처리량, 에러율, 자원 사용량
  conditions.csv    조건별(반복을 묶어) 평균과 최소~최대
  fig_*.png         보고서용 그래프 (단독 Baseline, 경합 영향)
  report.md         위 내용을 표로 정리한 문서

백분위는 부하 생성기가 요청마다 남긴 CSV에서 직접 계산한다 (대시보드의 히스토그램 추정치가 아님).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# ---------------------------------------------------------------- 그래프 스타일 (밝은 배경, 고정 색 순서)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # 1: p50/LLM, 2: p95/CV, 3: 세 번째 계열
REF = "#8a8984"  # 비교 기준선 (단독 실행)


def setup_style() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2, "lines.markersize": 6,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------- 데이터 읽기


def q(s: pd.Series, p: float) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(p / 100)) if len(s) else None


def mean(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def workload_stats(run_dir: Path, prefix: str, duration: float) -> dict:
    path = run_dir / "requests.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    m = df[df["phase"] == "measure"]
    ok = m[m["status"].astype(str) == "200"]
    # 처리량 = 측정 구간 "안에서 끝난" 성공 요청 수 / 구간 길이.
    # 보낸 요청은 줄을 서서라도 결국 끝나므로, 끝난 시각을 따지지 않으면 과부하에서도 처리량이 도착률만큼 나와 보인다.
    meta = json.loads((run_dir / "meta.json").read_text())
    w0 = float(meta["measure_start_unix"])
    w1 = w0 + duration
    done_at = pd.to_numeric(df["t_send_unix"], errors="coerce") + pd.to_numeric(df["latency_ms"], errors="coerce") / 1000
    completed = df[(df["status"].astype(str) == "200") & (done_at >= w0) & (done_at <= w1)]
    s = {
        "requests": len(m),
        "error_rate": round(1 - len(ok) / len(m), 4) if len(m) else None,
        "sent_rps": round(len(m) / duration, 3),
        "achieved_rps": round(len(completed) / duration, 3),
        "send_lag_p99_ms": q(m["send_lag_ms"], 99),
        "latency_p50_ms": q(ok["latency_ms"], 50),
        "latency_p95_ms": q(ok["latency_ms"], 95),
        "latency_p99_ms": q(ok["latency_ms"], 99),
    }
    if prefix == "llm":
        s.update({
            "ttft_p50_ms": q(ok["client_ttft_ms"], 50), "ttft_p95_ms": q(ok["client_ttft_ms"], 95),
            "ttft_p99_ms": q(ok["client_ttft_ms"], 99), "tpot_p50_ms": q(ok["gw_tpot_ms"], 50),
            "tpot_p95_ms": q(ok["gw_tpot_ms"], 95), "gw_queue_wait_p95_ms": q(ok["gw_queue_wait_ms"], 95),
            "tokens_per_s": round(pd.to_numeric(ok["output_tokens"], errors="coerce").sum() / duration, 1),
        })
    else:
        s.update({
            "queue_wait_p95_ms": q(ok["cv_queue_wait_ms"], 95), "batch_wait_p95_ms": q(ok["cv_batch_wait_ms"], 95),
            "inference_p50_ms": q(ok["cv_inference_ms"], 50), "decode_p50_ms": q(ok["cv_decode_ms"], 50),
            "batch_size_mean": mean(ok["cv_batch_size"]),
        })
    bad = m[m["status"].astype(str) != "200"]["status"].astype(str).value_counts()
    s["non200"] = ", ".join(f"{k}×{v}" for k, v in bad.items())
    return {f"{prefix}_{k}": v for k, v in s.items()}


def resource_stats(step_dir: Path) -> dict:
    path = step_dir / "resources.json"
    if not path.is_file():
        return {}
    r = json.loads(path.read_text())
    out = {}
    for name in ("gpu_util_pct", "gpu_mem_used_gb", "gpu_power_w", "host_cpu_pct", "llm_server_running",
                 "llm_server_waiting", "llm_kv_cache_pct", "gateway_llm_waiting", "cv_queue_depth", "cv_batch_size"):
        v = r.get(name) or {}
        out[f"res_{name}_mean"] = v.get("mean")
        out[f"res_{name}_max"] = v.get("max")
    for job, v in (r.get("cpu_cores_by_job") or {}).items():
        if isinstance(v, dict):
            out[f"res_cpu_cores_{job}_mean"] = v.get("mean")
    return out


def load(exp_dir: Path) -> tuple[pd.DataFrame, dict]:
    plan = json.loads((exp_dir / "plan.json").read_text())
    duration = float(plan["args"]["duration"])
    index = pd.read_csv(exp_dir / "runs.csv")
    rows = []
    for _, st in index.iterrows():
        step_dir = exp_dir / st["step_dir"]
        row = {k: st[k] for k in ("scenario", "rep", "llm_rate", "cv_rate", "status")}
        row.update(workload_stats(step_dir / "llm", "llm", duration))
        row.update(workload_stats(step_dir / "cv", "cv", duration))
        row.update(resource_stats(step_dir))
        rows.append(row)
    return pd.DataFrame(rows), plan


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """조건별로 반복을 묶는다: <지표> = 평균, <지표>__min / __max = 반복 간 범위."""
    keys = ["scenario", "llm_rate", "cv_rate"]
    num = [c for c in runs.columns if c not in keys + ["rep", "status"] and pd.api.types.is_numeric_dtype(runs[c])]
    g = runs[runs["status"] == "ok"].groupby(keys, dropna=False)[num]
    agg = g.mean()
    agg = agg.join(g.min().add_suffix("__min")).join(g.max().add_suffix("__max"))
    agg["reps"] = g.size()
    return agg.reset_index()


# ---------------------------------------------------------------- 그래프


def line(ax, df: pd.DataFrame, x: str, y: str, color: str, label: str, scale: float = 1.0) -> None:
    """반복 평균을 선으로, 반복 간 최소~최대를 수염으로."""
    d = df.dropna(subset=[y]).sort_values(x)
    if d.empty:
        return
    mid = d[y] * scale
    lo = (d[y] - d.get(f"{y}__min", d[y])) * scale
    hi = (d.get(f"{y}__max", d[y]) - d[y]) * scale
    ax.errorbar(d[x], mid, yerr=[lo.clip(lower=0), hi.clip(lower=0)], color=color, marker="o", capsize=3,
                elinewidth=1, label=label)


def ref_line(ax, value: float | None, label: str, scale: float = 1.0) -> None:
    if value is None or pd.isna(value):
        return
    ax.axhline(value * scale, color=REF, linestyle="--", linewidth=1.2)
    ax.annotate(label, xy=(1, value * scale), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)


def finish(fig, axes, path: Path, title: str) -> None:
    for ax in axes:
        ax.set_ylim(bottom=0)
        handles, _ = ax.get_legend_handles_labels()
        if len(handles) > 1:
            ax.legend(loc="best")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_baseline_llm(cond: pd.DataFrame, out: Path) -> Path | None:
    d = cond[cond["scenario"] == "llm-only"]
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    line(axes[0], d, "llm_rate", "llm_ttft_p50_ms", C1, "p50", 1 / 1000)
    line(axes[0], d, "llm_rate", "llm_ttft_p95_ms", C2, "p95", 1 / 1000)
    axes[0].set(title="첫 토큰 시간 (TTFT)", xlabel="LLM 도착률 (req/s)", ylabel="초")
    line(axes[1], d, "llm_rate", "llm_tpot_p50_ms", C1, "p50")
    line(axes[1], d, "llm_rate", "llm_tpot_p95_ms", C2, "p95")
    axes[1].set(title="토큰당 시간 (TPOT)", xlabel="LLM 도착률 (req/s)", ylabel="ms")
    lim = float(d["llm_rate"].max()) * 1.05
    axes[2].plot([0, lim], [0, lim], color=REF, linestyle="--", linewidth=1.2)
    axes[2].annotate("보낸 만큼 처리", xy=(lim * 0.5, lim * 0.5), xytext=(8, -10), textcoords="offset points",
                     fontsize=8, color=INK2)
    line(axes[2], d, "llm_rate", "llm_achieved_rps", C1, "처리량")
    axes[2].set(title="처리량 (측정 구간 안에 끝난 요청)", xlabel="LLM 도착률 (req/s)", ylabel="req/s")
    path = out / "fig_llm_baseline.png"
    finish(fig, axes, path, "LLM 단독: 도착률을 올릴 때")
    return path


def fig_baseline_cv(cond: pd.DataFrame, out: Path) -> Path | None:
    d = cond[cond["scenario"] == "cv-only"]
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    line(axes[0], d, "cv_rate", "cv_latency_p50_ms", C1, "p50")
    line(axes[0], d, "cv_rate", "cv_latency_p95_ms", C2, "p95")
    axes[0].set(title="전체 지연", xlabel="CV 도착률 (req/s)", ylabel="ms")
    line(axes[1], d, "cv_rate", "cv_batch_size_mean", C1, "평균 배치 크기")
    axes[1].set(title="평균 배치 크기", xlabel="CV 도착률 (req/s)", ylabel="이미지 수")
    lim = float(d["cv_rate"].max()) * 1.05
    axes[2].plot([0, lim], [0, lim], color=REF, linestyle="--", linewidth=1.2)
    axes[2].annotate("보낸 만큼 처리", xy=(lim * 0.5, lim * 0.5), xytext=(8, -10), textcoords="offset points",
                     fontsize=8, color=INK2)
    line(axes[2], d, "cv_rate", "cv_achieved_rps", C1, "처리량")
    axes[2].set(title="처리량 (측정 구간 안에 끝난 요청)", xlabel="CV 도착률 (req/s)", ylabel="req/s")
    path = out / "fig_cv_baseline.png"
    finish(fig, axes, path, "CV 단독: 도착률을 올릴 때")
    return path


def baseline_value(cond: pd.DataFrame, scenario: str, col: str, rate_col: str, rate: float):
    d = cond[(cond["scenario"] == scenario) & (cond[rate_col] == rate)]
    return None if d.empty or col not in d else d[col].iloc[0]


def resource_panel(ax, d: pd.DataFrame, x: str, xlabel: str) -> None:
    line(ax, d, x, "res_gpu_util_pct_mean", C1, "GPU 사용률")
    line(ax, d, x, "res_host_cpu_pct_mean", C2, "호스트 CPU")
    ax.set(title="자원 사용률 (측정 구간 평균)", xlabel=xlabel, ylabel="%")
    ax.set_ylim(0, 105)


def fig_contention_llm(cond: pd.DataFrame, plan: dict, out: Path) -> Path | None:
    d = cond[cond["scenario"] == "mixed-cv-sweep"]
    if d.empty:
        return None
    fixed = float(plan["args"]["llm_fixed"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    line(axes[0], d, "cv_rate", "llm_ttft_p95_ms", C2, "p95", 1 / 1000)
    ref_line(axes[0], baseline_value(cond, "llm-only", "llm_ttft_p95_ms", "llm_rate", fixed), "LLM 단독 p95", 1 / 1000)
    axes[0].set(title="LLM 첫 토큰 시간 p95", xlabel="함께 들어온 CV 도착률 (req/s)", ylabel="초")
    line(axes[1], d, "cv_rate", "llm_tpot_p50_ms", C1, "p50")
    ref_line(axes[1], baseline_value(cond, "llm-only", "llm_tpot_p50_ms", "llm_rate", fixed), "LLM 단독 p50")
    axes[1].set(title="LLM 토큰당 시간 p50", xlabel="함께 들어온 CV 도착률 (req/s)", ylabel="ms")
    resource_panel(axes[2], d, "cv_rate", "함께 들어온 CV 도착률 (req/s)")
    path = out / "fig_contention_llm.png"
    finish(fig, axes, path, f"CV 부하가 LLM에 주는 영향 (LLM {fixed:g} req/s 고정)")
    return path


def fig_contention_cv(cond: pd.DataFrame, plan: dict, out: Path) -> Path | None:
    d = cond[cond["scenario"] == "mixed-llm-sweep"]
    if d.empty:
        return None
    fixed = float(plan["args"]["cv_fixed"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    line(axes[0], d, "llm_rate", "cv_latency_p95_ms", C2, "p95")
    ref_line(axes[0], baseline_value(cond, "cv-only", "cv_latency_p95_ms", "cv_rate", fixed), "CV 단독 p95")
    axes[0].set(title="CV 전체 지연 p95", xlabel="함께 들어온 LLM 도착률 (req/s)", ylabel="ms")
    line(axes[1], d, "llm_rate", "cv_inference_p50_ms", C1, "p50")
    ref_line(axes[1], baseline_value(cond, "cv-only", "cv_inference_p50_ms", "cv_rate", fixed), "CV 단독 p50")
    axes[1].set(title="CV 추론 시간 p50 (배치당)", xlabel="함께 들어온 LLM 도착률 (req/s)", ylabel="ms")
    resource_panel(axes[2], d, "llm_rate", "함께 들어온 LLM 도착률 (req/s)")
    path = out / "fig_contention_cv.png"
    finish(fig, axes, path, f"LLM 부하가 CV에 주는 영향 (CV {fixed:g} req/s 고정)")
    return path


# ---------------------------------------------------------------- 문서


def fmt(v, digits: int = 0) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{v:,.{digits}f}"


def rng(row: pd.Series, col: str, scale: float = 1.0, digits: int = 0) -> str:
    """평균 (최소~최대). 반복이 1회면 평균만."""
    v = row.get(col)
    if v is None or pd.isna(v):
        return "-"
    lo, hi = row.get(f"{col}__min"), row.get(f"{col}__max")
    s = fmt(v * scale, digits)
    if row.get("reps", 1) > 1 and lo is not None and hi is not None and not pd.isna(lo) and hi != lo:
        s += f" ({fmt(lo * scale, digits)}~{fmt(hi * scale, digits)})"
    return s


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def knee(d: pd.DataFrame, rate_col: str, prefix: str, latency_col: str, latency_name: str) -> str:
    """포화가 시작된 첫 단계: 처리량이 실제로 보낸 양의 90% 아래로 떨어지거나, 지연 p95가 가장 낮은 도착률의 3배를 넘을 때."""
    d = d.sort_values(rate_col)
    base = d[latency_col].dropna()
    base = float(base.iloc[0]) if len(base) else None
    for _, r in d.iterrows():
        sent, done, lat = r.get(f"{prefix}_sent_rps"), r.get(f"{prefix}_achieved_rps"), r.get(latency_col)
        if sent and done is not None and not pd.isna(done) and done < 0.9 * sent:
            return (f"**{r[rate_col]:g} req/s에서 포화**: 측정 구간 안에 끝난 요청이 보낸 요청의 "
                    f"{done / sent:.0%}뿐 (보냄 {sent:.2f}, 처리 {done:.2f} req/s). 이 지점부터 줄이 계속 길어집니다.")
        if base and lat is not None and not pd.isna(lat) and lat > 3 * base:
            return (f"**{r[rate_col]:g} req/s에서 지연 급증**: {latency_name}가 가장 낮은 도착률의 "
                    f"{lat / base:.1f}배 ({base:,.0f} → {lat:,.0f} ms). 처리 한계에 가까워진 신호입니다.")
    return "측정한 범위 안에서는 포화 신호가 없습니다 (처리 한계는 더 높은 도착률에 있음)."


def write_report(cond: pd.DataFrame, runs: pd.DataFrame, plan: dict, figs: list[Path], out: Path) -> Path:
    a = plan["args"]
    lines = [
        f"# 실험 결과: {plan['name']}",
        "",
        f"- 시작: {plan['started_at']} · 커밋: `{plan['git_commit']}` · GPU: {plan['gpu']}",
        f"- 단계당 워밍업 {a['warmup']:g}초 + 측정 {a['duration']:g}초, 반복 {a['repeat']}회, 도착 간격 포아송",
        "- 표의 값은 반복 평균이고, 괄호는 반복 간 최소~최대입니다. 지연 백분위는 요청별 기록에서 직접 계산했습니다.",
        f"- 실패한 단계: {int((runs['status'] != 'ok').sum())}개",
        "",
    ]
    d = cond[cond["scenario"] == "llm-only"]
    if not d.empty:
        lines += ["## LLM 단독", "", knee(d, "llm_rate", "llm", "llm_ttft_p95_ms", "TTFT p95"), "", md_table(
            ["도착률 (req/s)", "처리량", "TTFT p50 (ms)", "TTFT p95 (ms)", "TPOT p50 (ms)", "e2e p95 (ms)",
             "에러율", "GPU %", "호스트 CPU %", "서버 처리 중 최대", "서버 내부 대기 최대", "KV cache 최대 %"],
            [[fmt(r["llm_rate"], 1), rng(r, "llm_achieved_rps", digits=2), rng(r, "llm_ttft_p50_ms"),
              rng(r, "llm_ttft_p95_ms"), rng(r, "llm_tpot_p50_ms", digits=1), rng(r, "llm_latency_p95_ms"),
              fmt(r.get("llm_error_rate"), 3), rng(r, "res_gpu_util_pct_mean"), rng(r, "res_host_cpu_pct_mean"),
              fmt(r.get("res_llm_server_running_max")), fmt(r.get("res_llm_server_waiting_max")),
              fmt(r.get("res_llm_kv_cache_pct_max"), 1)] for _, r in d.sort_values("llm_rate").iterrows()]), ""]
    d = cond[cond["scenario"] == "cv-only"]
    if not d.empty:
        lines += ["## CV 단독", "", knee(d, "cv_rate", "cv", "cv_latency_p95_ms", "전체 지연 p95"), "", md_table(
            ["도착률 (req/s)", "처리량", "p50 (ms)", "p95 (ms)", "p99 (ms)", "큐 대기 p95 (ms)", "추론 p50 (ms)",
             "해독 p50 (ms)", "평균 배치", "에러율", "GPU %", "호스트 CPU %"],
            [[fmt(r["cv_rate"], 1), rng(r, "cv_achieved_rps", digits=2), rng(r, "cv_latency_p50_ms"),
              rng(r, "cv_latency_p95_ms"), rng(r, "cv_latency_p99_ms"), rng(r, "cv_queue_wait_p95_ms"),
              rng(r, "cv_inference_p50_ms", digits=1), rng(r, "cv_decode_p50_ms", digits=1),
              rng(r, "cv_batch_size_mean", digits=2), fmt(r.get("cv_error_rate"), 3), rng(r, "res_gpu_util_pct_mean"),
              rng(r, "res_host_cpu_pct_mean")] for _, r in d.sort_values("cv_rate").iterrows()]), ""]
    d = cond[cond["scenario"] == "mixed-cv-sweep"]
    if not d.empty:
        base = cond[(cond["scenario"] == "llm-only") & (cond["llm_rate"] == float(a["llm_fixed"]))]
        rows = []
        if not base.empty:
            b = base.iloc[0]
            rows.append(["0 (LLM 단독)", rng(b, "llm_ttft_p95_ms"), rng(b, "llm_tpot_p50_ms", digits=1),
                         rng(b, "llm_latency_p95_ms"), "-", "-", "-", rng(b, "res_gpu_util_pct_mean"),
                         rng(b, "res_host_cpu_pct_mean")])
        rows += [[fmt(r["cv_rate"], 1), rng(r, "llm_ttft_p95_ms"), rng(r, "llm_tpot_p50_ms", digits=1),
                  rng(r, "llm_latency_p95_ms"), rng(r, "cv_achieved_rps", digits=2), rng(r, "cv_latency_p95_ms"),
                  fmt(r.get("cv_error_rate"), 3), rng(r, "res_gpu_util_pct_mean"), rng(r, "res_host_cpu_pct_mean")]
                 for _, r in d.sort_values("cv_rate").iterrows()]
        lines += [f"## 경합: LLM {float(a['llm_fixed']):g} req/s 고정 + CV 증가", "", md_table(
            ["CV 도착률", "LLM TTFT p95 (ms)", "LLM TPOT p50 (ms)", "LLM e2e p95 (ms)", "CV 처리량", "CV p95 (ms)",
             "CV 에러율", "GPU %", "호스트 CPU %"], rows), ""]
    d = cond[cond["scenario"] == "mixed-llm-sweep"]
    if not d.empty:
        base = cond[(cond["scenario"] == "cv-only") & (cond["cv_rate"] == float(a["cv_fixed"]))]
        rows = []
        if not base.empty:
            b = base.iloc[0]
            rows.append(["0 (CV 단독)", rng(b, "cv_latency_p95_ms"), rng(b, "cv_inference_p50_ms", digits=1),
                         rng(b, "cv_decode_p50_ms", digits=1), rng(b, "cv_queue_wait_p95_ms"),
                         fmt(b.get("cv_error_rate"), 3), "-", "-", rng(b, "res_gpu_util_pct_mean"),
                         rng(b, "res_host_cpu_pct_mean")])
        rows += [[fmt(r["llm_rate"], 1), rng(r, "cv_latency_p95_ms"), rng(r, "cv_inference_p50_ms", digits=1),
                  rng(r, "cv_decode_p50_ms", digits=1), rng(r, "cv_queue_wait_p95_ms"), fmt(r.get("cv_error_rate"), 3),
                  rng(r, "llm_ttft_p95_ms"), rng(r, "llm_achieved_rps", digits=2), rng(r, "res_gpu_util_pct_mean"),
                  rng(r, "res_host_cpu_pct_mean")] for _, r in d.sort_values("llm_rate").iterrows()]
        lines += [f"## 경합: CV {float(a['cv_fixed']):g} req/s 고정 + LLM 증가", "", md_table(
            ["LLM 도착률", "CV p95 (ms)", "CV 추론 p50 (ms)", "CV 해독 p50 (ms)", "CV 큐 대기 p95 (ms)", "CV 에러율",
             "LLM TTFT p95 (ms)", "LLM 처리량", "GPU %", "호스트 CPU %"], rows), ""]
    failed = []
    for _, r in runs.iterrows():
        for w in ("llm", "cv"):
            v = r.get(f"{w}_non200")
            if isinstance(v, str) and v:
                failed.append([r["scenario"], str(r["rep"]), fmt(r.get("llm_rate"), 1), fmt(r.get("cv_rate"), 1),
                               w.upper(), v])
    if failed:
        lines += ["## 200이 아닌 응답이 있었던 단계", "",
                  "측정 구간에 보낸 요청 중 실패한 것의 상태별 건수입니다. 504는 게이트웨이가 CV 결과를 기다리다 한도"
                  "(`CV_WAIT_TIMEOUT_S`)를 넘긴 것, 503은 LLM 동시성 대기 한도를 넘긴 것, timeout은 부하 생성기 쪽 한도입니다.",
                  "", md_table(["시나리오", "회차", "LLM 도착률", "CV 도착률", "워크로드", "상태별 건수"], failed), ""]
    lag = pd.concat([runs.get("llm_send_lag_p99_ms", pd.Series(dtype=float)),
                     runs.get("cv_send_lag_p99_ms", pd.Series(dtype=float))]).dropna()
    if len(lag) and lag.max() > 50:
        lines += [f"> 주의: 일부 단계에서 부하 생성기가 요청을 예정보다 늦게 보냈습니다 (send_lag p99 최대 {lag.max():.0f} ms)."
                  " 부하 생성기 자체가 바빴던 구간이라 해당 단계 해석에 주의하세요.", ""]
    if figs:
        lines += ["## 그래프", ""] + [f"![{f.stem}]({f.name})" for f in figs] + [""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dir", help="results/<name>")
    p.add_argument("--out", help="출력 폴더 (기본: <exp_dir>/report)")
    args = p.parse_args()
    exp_dir = Path(args.exp_dir)
    out = Path(args.out) if args.out else exp_dir / "report"
    out.mkdir(parents=True, exist_ok=True)

    setup_style()
    runs, plan = load(exp_dir)
    cond = aggregate(runs)
    runs.to_csv(out / "runs.csv", index=False)
    cond.to_csv(out / "conditions.csv", index=False)
    figs = [f for f in (fig_baseline_llm(cond, out), fig_baseline_cv(cond, out), fig_contention_llm(cond, plan, out),
                        fig_contention_cv(cond, plan, out)) if f]
    report = write_report(cond, runs, plan, figs, out)
    print(f"단계 {len(runs)}개, 조건 {len(cond)}개 → {out}/")
    for f in [out / "runs.csv", out / "conditions.csv", *figs, report]:
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
