#!/usr/bin/env python3
"""실험 결과 분석: results/<name>/ 을 읽어 표·그래프·요약 문서를 만든다.

  python analysis/report.py results/d3

만드는 것 (results/<name>/report/)
  runs.csv          단계×반복 1행: 지연 백분위, 처리량, goodput, 에러·거절률, 자원 사용량, 무효 사유
  conditions.csv    조건별(반복을 묶어) 평균과 최소~최대
  fig_*.png         보고서용 그래프 (단독 Baseline, 경합 영향)
  report.md         위 내용을 표로 정리한 문서 (H3: 게이트웨이 밖에서 쓴 시간, 무효 단계 목록 포함)

백분위는 부하 생성기가 요청마다 남긴 CSV에서 직접 계산한다 (대시보드의 히스토그램 추정치가 아님).
여러 실험의 전후 비교는 analysis/compare.py.
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


# SLO 후보 (main()에서 --slo-* 옵션으로 바꿀 수 있음). goodput 계산에 쓴다.
SLO = {"ttft_ms": 1000.0, "tpot_ms": 50.0, "cv_ms": 500.0}
# 측정이 믿을 만한지: 부하 생성기가 요청을 못 보냈거나(dropped), 예정보다 이만큼 넘게 늦게 보냈으면 무효로 표시
SEND_LAG_LIMIT_MS = 200.0
REJECT_STATUSES = ("429", "503")  # 설계된 거절: 큐 가득 참(429), 대기 한도·일시적 과부하(503)
# d3까지 쓰던 시나리오 이름 → 지금 이름 (experiments/run.py 와 같은 표). 옛 결과도 새 이름으로 읽는다
SCENARIO_ALIASES = {"mixed-cv-sweep": "contention-cv-sweep", "mixed-llm-sweep": "contention-llm-sweep"}


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(index=df.index, dtype=float)


def workload_stats(run_dir: Path, prefix: str, duration: float) -> dict:
    path = run_dir / "requests.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    status = df["status"].astype(str)
    m = df[df["phase"] == "measure"]
    m_status = status[m.index]
    ok = m[m_status == "200"]
    # 처리량 = 측정 구간 "안에서 끝난" 성공 요청 수 / 구간 길이.
    # 보낸 요청은 줄을 서서라도 결국 끝나므로, 끝난 시각을 따지지 않으면 과부하에서도 처리량이 도착률만큼 나와 보인다.
    meta = json.loads((run_dir / "meta.json").read_text())
    w0 = float(meta["measure_start_unix"])
    w1 = w0 + duration
    done_at = num(df, "t_send_unix") + num(df, "latency_ms") / 1000
    in_window = (status == "200") & (done_at >= w0) & (done_at <= w1)
    completed = df[in_window]
    # goodput = 그중 SLO까지 지킨 요청만. 일부를 거절하는 최적화는 처리량이 아니라 이 값으로 비교해야 한다.
    if prefix == "llm":
        meets = (num(df, "client_ttft_ms") <= SLO["ttft_ms"]) & (num(df, "gw_tpot_ms").fillna(0) <= SLO["tpot_ms"])
    else:
        meets = num(df, "latency_ms") <= SLO["cv_ms"]
    # 게이트웨이 밖에서 쓴 시간 = 클라이언트가 잰 지연 - 게이트웨이가 잰 지연 (요청마다 빼서 백분위). H3 확인용.
    outside = (num(ok, "latency_ms") - num(ok, "gw_e2e_ms")).dropna()
    n = len(m)
    s = {
        "requests": n,
        "error_rate": round(1 - len(ok) / n, 4) if n else None,
        "reject_rate": round(m_status.isin(REJECT_STATUSES).sum() / n, 4) if n else None,
        "dropped": int((m_status == "dropped").sum()),
        "sent_rps": round(n / duration, 3),
        "achieved_rps": round(len(completed) / duration, 3),
        "goodput_rps": round(int((in_window & meets).sum()) / duration, 3),
        "send_lag_p99_ms": q(m["send_lag_ms"], 99) if "send_lag_ms" in m else None,
        "latency_p50_ms": q(ok["latency_ms"], 50),
        "latency_p95_ms": q(ok["latency_ms"], 95),
        "latency_p99_ms": q(ok["latency_ms"], 99),
        "gw_e2e_p95_ms": q(ok["gw_e2e_ms"], 95) if "gw_e2e_ms" in ok else None,
        "outside_gw_p50_ms": q(outside, 50),
        "outside_gw_p95_ms": q(outside, 95),
    }
    if prefix == "llm":
        s.update({
            "ttft_p50_ms": q(ok["client_ttft_ms"], 50), "ttft_p95_ms": q(ok["client_ttft_ms"], 95),
            "ttft_p99_ms": q(ok["client_ttft_ms"], 99), "tpot_p50_ms": q(ok["gw_tpot_ms"], 50),
            "tpot_p95_ms": q(ok["gw_tpot_ms"], 95), "gw_queue_wait_p95_ms": q(ok["gw_queue_wait_ms"], 95),
            # 출력 토큰 처리량 = 측정 구간 안에 끝난 성공 요청의 출력 토큰 합 / 구간 길이 (LLM이 GPU에 시킨 일의 양)
            "tokens_per_s": round(float(num(completed, "output_tokens").sum()) / duration, 1),
        })
    else:
        # 이미지당 시간 = 요청마다 (그 요청이 속한 배치의 시간 ÷ 배치 크기). 배치를 키우는 실험은 배치당이 아니라 이 값으로 비교
        bsz = num(ok, "cv_batch_size").where(lambda x: x > 0)
        s.update({
            "queue_wait_p95_ms": q(ok["cv_queue_wait_ms"], 95), "batch_wait_p95_ms": q(ok["cv_batch_wait_ms"], 95),
            "inference_p50_ms": q(ok["cv_inference_ms"], 50), "decode_p50_ms": q(ok["cv_decode_ms"], 50),
            "inference_per_img_p50_ms": q(num(ok, "cv_inference_ms") / bsz, 50),
            "decode_per_img_p50_ms": q(num(ok, "cv_decode_ms") / bsz, 50),
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
        row["scenario"] = SCENARIO_ALIASES.get(row["scenario"], row["scenario"])
        row.update(workload_stats(step_dir / "llm", "llm", duration))
        row.update(workload_stats(step_dir / "cv", "cv", duration))
        row.update(resource_stats(step_dir))
        row["invalid_reason"] = invalid_reason(row)
        rows.append(row)
    return pd.DataFrame(rows), plan


def invalid_reason(row: dict) -> str:
    """부하 생성기가 계획대로 요청을 보내지 못한 단계. 시스템이 아니라 측정 도구의 한계라 용량 판단에서 뺀다."""
    reasons = []
    for w in ("llm", "cv"):
        dropped = row.get(f"{w}_dropped") or 0
        lag = row.get(f"{w}_send_lag_p99_ms")
        if dropped:
            reasons.append(f"{w.upper()} 못 보낸 요청 {int(dropped)}건")
        if lag is not None and not pd.isna(lag) and lag > SEND_LAG_LIMIT_MS:
            reasons.append(f"{w.upper()} 보내는 시각 지연 p99 {lag:,.0f} ms")
    return ", ".join(reasons)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """조건별로 반복을 묶는다: <지표> = 평균, <지표>__min / __max = 반복 간 범위."""
    keys = ["scenario", "llm_rate", "cv_rate"]
    skip = keys + ["rep", "status", "invalid_reason"]
    cols = [c for c in runs.columns if c not in skip and pd.api.types.is_numeric_dtype(runs[c])]
    ok = runs[runs["status"] == "ok"].copy()
    ok["invalid"] = (ok["invalid_reason"].fillna("") != "").astype(int)
    g = ok.groupby(keys, dropna=False)
    agg = g[cols].mean()
    agg = agg.join(g[cols].min().add_suffix("__min")).join(g[cols].max().add_suffix("__max"))
    agg["reps"] = g.size()
    agg["invalid_reps"] = g["invalid"].sum()
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
    d = cond[cond["scenario"] == "contention-cv-sweep"]
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
    d = cond[cond["scenario"] == "contention-llm-sweep"]
    if d.empty:
        return None
    fixed = float(plan["args"]["cv_fixed"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    line(axes[0], d, "llm_rate", "cv_latency_p95_ms", C2, "p95")
    ref_line(axes[0], baseline_value(cond, "cv-only", "cv_latency_p95_ms", "cv_rate", fixed), "CV 단독 p95")
    axes[0].set(title="CV 전체 지연 p95", xlabel="함께 들어온 LLM 도착률 (req/s)", ylabel="ms")
    line(axes[1], d, "llm_rate", "cv_inference_p50_ms", C1, "배치당")
    line(axes[1], d, "llm_rate", "cv_inference_per_img_p50_ms", C3, "이미지당")
    ref_line(axes[1], baseline_value(cond, "cv-only", "cv_inference_p50_ms", "cv_rate", fixed), "CV 단독 배치당")
    axes[1].set(title="CV 추론 시간 p50", xlabel="함께 들어온 LLM 도착률 (req/s)", ylabel="ms")
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
    if row.get("reps", 1) > 1 and lo is not None and hi is not None and not pd.isna(lo):
        a, b = fmt(lo * scale, digits), fmt(hi * scale, digits)
        if a != b:
            s += f" ({a}~{b})"
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


def rate_label(r: pd.Series, col: str) -> str:
    """도착률. 반복 중 하나라도 측정이 무효였으면 † 표시."""
    s = fmt(r[col], 1)
    return s + " †" if r.get("invalid_reps", 0) else s


def write_report(cond: pd.DataFrame, runs: pd.DataFrame, plan: dict, figs: list[Path], out: Path) -> Path:
    a = plan["args"]
    lines = [
        f"# 실험 결과: {plan['name']}",
        "",
        f"- 시작: {plan['started_at']} · 커밋: `{plan['git_commit']}` · GPU: {plan['gpu']}",
        f"- 단계당 워밍업 {a['warmup']:g}초 + 측정 {a['duration']:g}초, 반복 {a['repeat']}회, 도착 간격 포아송",
        "- 표의 값은 반복 평균이고, 괄호는 반복 간 최소~최대입니다. 지연 백분위는 요청별 기록에서 직접 계산했습니다.",
        "- CV 추론·디코딩의 '배치당'은 배치 한 번에 걸린 시간, '/이미지'는 요청마다 (배치 시간 ÷ 배치 크기)의 p50입니다. "
        "디코딩은 JPEG를 픽셀 배열로 푸는 CPU 작업입니다. 출력 tok/s = 측정 구간 안에 끝난 LLM 요청의 출력 토큰 합 ÷ 구간 길이.",
        f"- goodput = 측정 구간 안에 끝나고 SLO까지 지킨 성공 요청 수/초 (LLM: TTFT ≤ {SLO['ttft_ms']:g} ms 그리고 "
        f"TPOT ≤ {SLO['tpot_ms']:g} ms, CV: 전체 지연 ≤ {SLO['cv_ms']:g} ms)",
        "- 도착률 옆 †: 부하 생성기가 계획대로 보내지 못한 회차가 있는 단계 (아래 '측정이 무효였던 단계' 참고)",
        f"- 실패한 단계: {int((runs['status'] != 'ok').sum())}개",
        "",
    ]
    d = cond[cond["scenario"] == "llm-only"]
    if not d.empty:
        lines += ["## LLM 단독", "", knee(d, "llm_rate", "llm", "llm_ttft_p95_ms", "TTFT p95"), "", md_table(
            ["도착률 (req/s)", "처리량", "goodput", "출력 tok/s", "TTFT p50 (ms)", "TTFT p95 (ms)", "TPOT p50 (ms)",
             "TPOT p95 (ms)", "e2e p95 (ms)", "에러율", "GPU %", "호스트 CPU %", "서버 처리 중 최대", "서버 내부 대기 최대",
             "KV cache 최대 %"],
            [[rate_label(r, "llm_rate"), rng(r, "llm_achieved_rps", digits=2), rng(r, "llm_goodput_rps", digits=2),
              rng(r, "llm_tokens_per_s"),
              rng(r, "llm_ttft_p50_ms"), rng(r, "llm_ttft_p95_ms"), rng(r, "llm_tpot_p50_ms", digits=1),
              rng(r, "llm_tpot_p95_ms", digits=1), rng(r, "llm_latency_p95_ms"), fmt(r.get("llm_error_rate"), 3),
              rng(r, "res_gpu_util_pct_mean"), rng(r, "res_host_cpu_pct_mean"),
              fmt(r.get("res_llm_server_running_max")), fmt(r.get("res_llm_server_waiting_max")),
              fmt(r.get("res_llm_kv_cache_pct_max"), 1)] for _, r in d.sort_values("llm_rate").iterrows()]), ""]
    d = cond[cond["scenario"] == "cv-only"]
    if not d.empty:
        lines += ["## CV 단독", "", knee(d, "cv_rate", "cv", "cv_latency_p95_ms", "전체 지연 p95"), "", md_table(
            ["도착률 (req/s)", "처리량", "goodput", "p50 (ms)", "p95 (ms)", "p99 (ms)", "큐 대기 p95 (ms)",
             "큐 길이 최대", "추론 p50 (ms)", "추론/이미지 p50 (ms)", "디코딩 p50 (ms)", "디코딩/이미지 p50 (ms)",
             "평균 배치", "에러율", "거절률", "GPU %", "호스트 CPU %"],
            [[rate_label(r, "cv_rate"), rng(r, "cv_achieved_rps", digits=2), rng(r, "cv_goodput_rps", digits=2),
              rng(r, "cv_latency_p50_ms"), rng(r, "cv_latency_p95_ms"), rng(r, "cv_latency_p99_ms"),
              rng(r, "cv_queue_wait_p95_ms"), fmt(r.get("res_cv_queue_depth_max__max")),
              rng(r, "cv_inference_p50_ms", digits=1), rng(r, "cv_inference_per_img_p50_ms", digits=1),
              rng(r, "cv_decode_p50_ms", digits=1), rng(r, "cv_decode_per_img_p50_ms", digits=1),
              rng(r, "cv_batch_size_mean", digits=2), fmt(r.get("cv_error_rate"), 3), fmt(r.get("cv_reject_rate"), 3),
              rng(r, "res_gpu_util_pct_mean"), rng(r, "res_host_cpu_pct_mean")]
             for _, r in d.sort_values("cv_rate").iterrows()]), ""]
    d = cond[cond["scenario"] == "contention-cv-sweep"]
    if not d.empty:
        base = cond[(cond["scenario"] == "llm-only") & (cond["llm_rate"] == float(a["llm_fixed"]))]
        rows = []
        if not base.empty:
            b = base.iloc[0]
            rows.append(["0 (LLM 단독)", rng(b, "llm_ttft_p95_ms"), rng(b, "llm_tpot_p50_ms", digits=1)]
                        + ["-"] * 9 + [rng(b, "res_host_cpu_pct_mean")])
        rows += [[rate_label(r, "cv_rate"), rng(r, "llm_ttft_p95_ms"), rng(r, "llm_tpot_p50_ms", digits=1),
                  rng(r, "cv_achieved_rps", digits=2), rng(r, "cv_goodput_rps", digits=2), rng(r, "cv_latency_p95_ms"),
                  rng(r, "cv_inference_p50_ms", digits=1), rng(r, "cv_inference_per_img_p50_ms", digits=1),
                  rng(r, "cv_batch_size_mean", digits=2),
                  fmt(r.get("res_cv_queue_depth_max__max")), fmt(r.get("cv_error_rate"), 3),
                  fmt(r.get("cv_reject_rate"), 3), rng(r, "res_host_cpu_pct_mean")]
                 for _, r in d.sort_values("cv_rate").iterrows()]
        lines += [f"## 경합: LLM {float(a['llm_fixed']):g} req/s 고정 + CV 증가", "", md_table(
            ["CV 도착률", "LLM TTFT p95 (ms)", "LLM TPOT p50 (ms)", "CV 처리량", "CV goodput", "CV p95 (ms)",
             "CV 추론 p50 (ms)", "CV 추론/이미지 p50 (ms)", "CV 평균 배치", "CV 큐 길이 최대", "CV 에러율", "CV 거절률",
             "호스트 CPU %"], rows),
                  "", "GPU 사용률은 LLM이 돌고 있으면 항상 100% 근처라 이 표에서는 뺐습니다.", ""]
    d = cond[cond["scenario"] == "contention-llm-sweep"]
    if not d.empty:
        base = cond[(cond["scenario"] == "cv-only") & (cond["cv_rate"] == float(a["cv_fixed"]))]
        rows = []
        if not base.empty:
            b = base.iloc[0]
            rows.append(["0 (CV 단독)", rng(b, "cv_latency_p95_ms"), rng(b, "cv_inference_p50_ms", digits=1),
                         rng(b, "cv_inference_per_img_p50_ms", digits=1), rng(b, "cv_decode_p50_ms", digits=1),
                         rng(b, "cv_queue_wait_p95_ms"), fmt(b.get("cv_error_rate"), 3), "-", "-", "-", "-",
                         rng(b, "res_host_cpu_pct_mean")])
        rows += [[rate_label(r, "llm_rate"), rng(r, "cv_latency_p95_ms"), rng(r, "cv_inference_p50_ms", digits=1),
                  rng(r, "cv_inference_per_img_p50_ms", digits=1), rng(r, "cv_decode_p50_ms", digits=1),
                  rng(r, "cv_queue_wait_p95_ms"), fmt(r.get("cv_error_rate"), 3),
                  rng(r, "llm_ttft_p95_ms"), rng(r, "llm_tpot_p50_ms", digits=1), rng(r, "llm_achieved_rps", digits=2),
                  rng(r, "llm_tokens_per_s"), rng(r, "res_host_cpu_pct_mean")]
                 for _, r in d.sort_values("llm_rate").iterrows()]
        lines += [f"## 경합: CV {float(a['cv_fixed']):g} req/s 고정 + LLM 증가", "", md_table(
            ["LLM 도착률", "CV p95 (ms)", "CV 추론 p50 (ms)", "CV 추론/이미지 p50 (ms)", "CV 디코딩 p50 (ms)",
             "CV 큐 대기 p95 (ms)", "CV 에러율", "LLM TTFT p95 (ms)", "LLM TPOT p50 (ms)", "LLM 처리량",
             "LLM 출력 tok/s", "호스트 CPU %"], rows), ""]
    lines += h3_section(cond)
    failed = []
    for _, r in runs.iterrows():
        for w in ("llm", "cv"):
            v = r.get(f"{w}_non200")
            if isinstance(v, str) and v:
                failed.append([r["scenario"], str(r["rep"]), fmt(r.get("llm_rate"), 1), fmt(r.get("cv_rate"), 1),
                               w.upper(), v])
    if failed:
        lines += ["## 200이 아닌 응답이 있었던 단계", "",
                  "측정 구간에 보낸 요청 중 실패한 것의 상태별 건수입니다. 429는 CV 큐가 가득 차 거절한 것(설계된 거절), "
                  "503은 LLM 동시성 대기 한도 초과 또는 게이트웨이의 Redis 연결 부족, 504는 CV 결과 대기 한도"
                  "(`CV_WAIT_TIMEOUT_S`) 초과, 500은 처리되지 않은 오류입니다. dropped·timeout·conn_error는 부하 생성기 쪽에서 "
                  "기록한 것입니다.",
                  "", md_table(["시나리오", "회차", "LLM 도착률", "CV 도착률", "워크로드", "상태별 건수"], failed), ""]
    bad = runs[runs["invalid_reason"].fillna("") != ""]
    if len(bad):
        lines += ["## 측정이 무효였던 단계", "",
                  f"부하 생성기가 요청을 다 보내지 못했거나(dropped) 예정보다 {SEND_LAG_LIMIT_MS:g} ms 넘게 늦게 보낸 회차입니다. "
                  "시스템이 아니라 측정 도구가 먼저 한계에 닿은 것이라, 용량 판단에서는 빼고 '과부하 때 어떻게 실패하는가'를 "
                  "보는 자료로만 씁니다.", "",
                  md_table(["시나리오", "회차", "LLM 도착률", "CV 도착률", "이유"],
                           [[r["scenario"], str(r["rep"]), fmt(r.get("llm_rate"), 1), fmt(r.get("cv_rate"), 1),
                             r["invalid_reason"]] for _, r in bad.iterrows()]), ""]
    if figs:
        lines += ["## 그래프", ""] + [f"![{f.stem}]({f.name})" for f in figs] + [""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path


def h3_section(cond: pd.DataFrame) -> list[str]:
    """H3: 게이트웨이(파이썬 프로세스 하나)와 부하 생성기가 지연을 얼마나 더하는가."""
    rows = []
    for scenario, w, rate_col in (("llm-only", "llm", "llm_rate"), ("cv-only", "cv", "cv_rate"),
                                  ("contention-cv-sweep", "cv", "cv_rate")):
        d = cond[cond["scenario"] == scenario].sort_values(rate_col)
        for _, r in d.iterrows():
            if pd.isna(r.get(f"{w}_outside_gw_p95_ms", float("nan"))):
                continue
            rows.append([scenario, w.upper(), rate_label(r, rate_col), rng(r, f"{w}_latency_p95_ms"),
                         rng(r, f"{w}_gw_e2e_p95_ms"), rng(r, f"{w}_outside_gw_p50_ms"),
                         rng(r, f"{w}_outside_gw_p95_ms"), rng(r, "res_cpu_cores_gateway_mean", digits=2),
                         rng(r, f"{w}_send_lag_p99_ms")])
    if not rows:
        return []
    return ["## 게이트웨이 밖에서 쓴 시간 (H3 확인)", "",
            "클라이언트가 잰 지연에서 게이트웨이가 잰 지연을 요청마다 뺀 값입니다. 요청을 받아 처리 함수가 시작되기 전"
            "(HTTP 수신, 이미지 업로드 파싱, 이벤트 루프 대기)과 부하 생성기 자신의 지연이 여기에 들어갑니다. 부하가 오를수록 "
            "이 값이 커지고 게이트웨이 CPU가 1코어에 가까워지면, 파이썬 프로세스 하나인 게이트웨이가 병목이라는 뜻(H3)입니다. "
            "부하 생성기의 보내는 시각 지연(send_lag)도 함께 커지면 부하 생성기 탓일 수 있습니다.", "",
            md_table(["시나리오", "워크로드", "도착률", "클라이언트 p95 (ms)", "게이트웨이 p95 (ms)", "밖 p50 (ms)",
                      "밖 p95 (ms)", "게이트웨이 CPU (코어)", "send_lag p99 (ms)"], rows), ""]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dir", help="results/<name>")
    p.add_argument("--out", help="출력 폴더 (기본: <exp_dir>/report)")
    p.add_argument("--slo-ttft-ms", type=float, default=SLO["ttft_ms"], help="goodput 기준: LLM TTFT 상한")
    p.add_argument("--slo-tpot-ms", type=float, default=SLO["tpot_ms"], help="goodput 기준: LLM TPOT 상한")
    p.add_argument("--slo-cv-ms", type=float, default=SLO["cv_ms"], help="goodput 기준: CV 전체 지연 상한")
    args = p.parse_args()
    SLO.update(ttft_ms=args.slo_ttft_ms, tpot_ms=args.slo_tpot_ms, cv_ms=args.slo_cv_ms)
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
