#!/usr/bin/env python3
"""여러 실험의 전후 비교: 같은 시나리오·같은 도착률끼리 나란히 놓는다.

  python analysis/compare.py results/e2-base results/e2a-batch16 results/e2b-workers2
  make compare EXPS="e2-base e2a-batch16 e2b-workers2"

첫 번째 실험을 기준(최적화 전)으로 보고, 나머지를 그 옆에 놓는다.

만드는 것 (기본: results/compare/<이름들>/)
  compare.md     바꾼 설정, 실험별 용량 요약, 시나리오별 비교표
  compare.csv    비교에 쓴 조건별 값 전체
  fig_compare_<시나리오>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as rp  # noqa: E402

COLORS = [rp.REF, rp.C1, rp.C2, rp.C3, "#7b5cd6", "#c9a227"]  # 기준은 회색, 나머지는 고정 순서

# 시나리오별: (도착률 열, 도착률 이름, [(열, 표 이름, 자릿수)], 그래프 패널 [(열, 제목, y 이름, 로그축)])
SPECS = {
    "llm-only": ("llm_rate", "LLM 도착률", [
        ("llm_achieved_rps", "처리량", 2), ("llm_goodput_rps", "goodput", 2), ("llm_tokens_per_s", "출력 tok/s", 0),
        ("llm_ttft_p95_ms", "TTFT p95 (ms)", 0), ("llm_tpot_p50_ms", "TPOT p50 (ms)", 1), ("llm_tpot_p95_ms", "TPOT p95 (ms)", 1),
        ("res_llm_server_running_max", "서버 처리 중 최대", 0), ("res_llm_kv_cache_pct_max", "KV cache 최대 %", 1),
    ], [("llm_goodput_rps", "goodput (SLO 안에 끝난 요청)", "req/s", False),
        ("llm_ttft_p95_ms", "TTFT p95", "ms", True), ("llm_tpot_p50_ms", "TPOT p50 (잃는 것)", "ms", False)]),
    "cv-only": ("cv_rate", "CV 도착률", [
        ("cv_achieved_rps", "처리량", 2), ("cv_goodput_rps", "goodput", 2), ("cv_latency_p95_ms", "p95 (ms)", 0),
        ("cv_inference_p50_ms", "추론 p50 (ms)", 1), ("cv_inference_per_img_p50_ms", "추론/이미지 p50 (ms)", 1),
        ("cv_decode_per_img_p50_ms", "디코딩/이미지 p50 (ms)", 1), ("cv_batch_size_mean", "평균 배치", 2),
        ("cv_error_rate", "에러율", 3), ("cv_reject_rate", "거절률", 3), ("res_gpu_mem_used_gb_max", "GPU 메모리 최대 (GB)", 1),
    ], [("cv_goodput_rps", "goodput (SLO 안에 끝난 요청)", "req/s", False),
        ("cv_latency_p95_ms", "CV 전체 지연 p95", "ms", True), ("cv_reject_rate", "거절률", "비율", False)]),
    "contention-cv-sweep": ("cv_rate", "CV 도착률", [
        ("cv_goodput_rps", "CV goodput", 2), ("cv_latency_p95_ms", "CV p95 (ms)", 0),
        ("cv_inference_p50_ms", "CV 추론 p50 (ms)", 1), ("cv_inference_per_img_p50_ms", "CV 추론/이미지 p50 (ms)", 1),
        ("cv_batch_size_mean", "CV 평균 배치", 2),
        ("cv_error_rate", "CV 에러율", 3), ("cv_reject_rate", "CV 거절률", 3), ("llm_ttft_p95_ms", "LLM TTFT p95 (ms)", 0),
        ("llm_tpot_p50_ms", "LLM TPOT p50 (ms)", 1), ("res_gpu_mem_used_gb_max", "GPU 메모리 최대 (GB)", 1),
        ("res_host_cpu_pct_mean", "호스트 CPU %", 0),
    ], [("cv_goodput_rps", "CV goodput (SLO 안에 끝난 요청)", "req/s", False),
        ("cv_latency_p95_ms", "CV 전체 지연 p95", "ms", True), ("llm_tpot_p50_ms", "LLM TPOT p50 (잃는 것)", "ms", False)]),
    "contention-llm-sweep": ("llm_rate", "LLM 도착률", [
        ("cv_latency_p95_ms", "CV p95 (ms)", 0), ("cv_inference_p50_ms", "CV 추론 p50 (ms)", 1),
        ("cv_inference_per_img_p50_ms", "CV 추론/이미지 p50 (ms)", 1), ("cv_error_rate", "CV 에러율", 3),
        ("llm_achieved_rps", "LLM 처리량", 2), ("llm_tokens_per_s", "LLM 출력 tok/s", 0),
        ("llm_ttft_p95_ms", "LLM TTFT p95 (ms)", 0),
        ("llm_tpot_p50_ms", "LLM TPOT p50 (ms)", 1), ("res_gpu_mem_used_gb_max", "GPU 메모리 최대 (GB)", 1),
    ], [("cv_latency_p95_ms", "CV 전체 지연 p95", "ms", True), ("cv_inference_per_img_p50_ms", "CV 추론 p50 (이미지당)", "ms", False),
        ("llm_tpot_p50_ms", "LLM TPOT p50", "ms", False)]),
}


def read_env(exp_dir: Path) -> dict[str, str]:
    path = exp_dir / "env.txt"
    out: dict[str, str] = {}
    if path.is_file():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                out[k.strip()] = v.split(" #")[0].strip()
    return out


def env_diff(labels: list[str], envs: list[dict[str, str]]) -> str:
    keys = sorted({k for e in envs for k in e})
    rows = [[k] + [e.get(k, "-") for e in envs] for k in keys if len({e.get(k) for e in envs}) > 1]
    if not rows:
        return "실험 사이에 다른 `.env` 설정이 없습니다 (같은 설정의 반복 실험이거나 env.txt가 없음)."
    return rp.md_table(["설정"] + labels, rows)


def meets_slo(r: pd.Series, scenario: str) -> bool:
    if r.get("invalid_reps", 0):
        return False
    if scenario in ("llm-only",):
        ttft, tpot = r.get("llm_ttft_p95_ms"), r.get("llm_tpot_p95_ms")
        lat_ok = ttft is not None and not pd.isna(ttft) and ttft <= rp.SLO["ttft_ms"] and (
            tpot is None or pd.isna(tpot) or tpot <= rp.SLO["tpot_ms"])
        err = r.get("llm_error_rate")
    else:
        p95 = r.get("cv_latency_p95_ms")
        lat_ok = p95 is not None and not pd.isna(p95) and p95 <= rp.SLO["cv_ms"]
        err = r.get("cv_error_rate")
    return bool(lat_ok and err is not None and not pd.isna(err) and err <= 0.01)


def capacity_rows(cond: pd.DataFrame, labels: list[str]) -> list[list[str]]:
    """실험×시나리오별: 모든 요청을 받아들이면서 SLO를 지킨 최대 도착률, 그리고 최대 goodput."""
    rows = []
    for scenario, (rate_col, _, _, _) in SPECS.items():
        target = "llm" if rate_col == "llm_rate" else "cv"  # 도착률을 올린 쪽의 용량을 본다
        for label in labels:
            d = cond[(cond["scenario"] == scenario) & (cond["exp"] == label)].sort_values(rate_col)
            if d.empty:
                continue
            ok = [r[rate_col] for _, r in d.iterrows() if meets_slo(r, "llm-only" if target == "llm" else "cv")]
            gp_col = f"{target}_goodput_rps"
            gp = d[gp_col].max() if gp_col in d else None
            at = d.loc[d[gp_col].idxmax(), rate_col] if gp_col in d and d[gp_col].notna().any() else None
            rows.append([scenario, label, rp.fmt(max(ok), 1) if ok else "측정 범위 아래",
                         rp.fmt(gp, 2), rp.fmt(at, 1), target.upper()])
    return rows


def compare_table(cond: pd.DataFrame, scenario: str, labels: list[str]) -> str:
    rate_col, rate_name, cols, _ = SPECS[scenario]
    d = cond[cond["scenario"] == scenario]
    rows = []
    for rate in sorted(d[rate_col].dropna().unique()):
        for label in labels:
            hit = d[(d[rate_col] == rate) & (d["exp"] == label)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append([rp.rate_label(r, rate_col), label] + [rp.rng(r, c, digits=dg) for c, _, dg in cols])
    return rp.md_table([rate_name, "실험"] + [name for _, name, _ in cols], rows)


def compare_fig(cond: pd.DataFrame, scenario: str, labels: list[str], out: Path) -> Path | None:
    rate_col, rate_name, _, panels = SPECS[scenario]
    d = cond[cond["scenario"] == scenario]
    if d["exp"].nunique() < 2:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8))
    for ax, (col, title, unit, log) in zip(axes, panels):
        for i, label in enumerate(labels):
            sub = d[d["exp"] == label]
            if col in sub:
                rp.line(ax, sub, rate_col, col, COLORS[i % len(COLORS)], label)
        ax.set(title=title, xlabel=f"{rate_name} (req/s)", ylabel=unit)
        if log:
            ax.set_yscale("log")
            if col.startswith("cv_latency"):
                rp.ref_line(ax, rp.SLO["cv_ms"], f"SLO {rp.SLO['cv_ms']:g} ms")
            elif col.startswith("llm_ttft"):
                rp.ref_line(ax, rp.SLO["ttft_ms"], f"SLO {rp.SLO['ttft_ms']:g} ms")
    for ax, (_, _, _, log) in zip(axes, panels):
        if not log:
            ax.set_ylim(bottom=0)
        handles, _ = ax.get_legend_handles_labels()
        if len(handles) > 1:
            ax.legend(loc="best")
    fig.suptitle(f"전후 비교: {scenario}", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=rp.INK)
    fig.tight_layout()
    path = out / f"fig_compare_{scenario}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dirs", nargs="+", help="results/<이름> 여러 개. 첫 번째가 기준")
    p.add_argument("--labels", help="쉼표로 구분한 표시 이름 (기본: 폴더 이름)")
    p.add_argument("--out", help="출력 폴더 (기본: results/compare/<이름들>)")
    p.add_argument("--slo-ttft-ms", type=float, default=rp.SLO["ttft_ms"])
    p.add_argument("--slo-tpot-ms", type=float, default=rp.SLO["tpot_ms"])
    p.add_argument("--slo-cv-ms", type=float, default=rp.SLO["cv_ms"])
    args = p.parse_args()
    if len(args.exp_dirs) < 2:
        p.error("비교할 실험을 두 개 이상 주세요")
    rp.SLO.update(ttft_ms=args.slo_ttft_ms, tpot_ms=args.slo_tpot_ms, cv_ms=args.slo_cv_ms)

    dirs = [Path(x) for x in args.exp_dirs]
    labels = args.labels.split(",") if args.labels else [d.name for d in dirs]
    if len(labels) != len(dirs):
        p.error("--labels 개수가 실험 개수와 다릅니다")
    out = Path(args.out) if args.out else Path("results") / "compare" / "__".join(d.name for d in dirs)
    out.mkdir(parents=True, exist_ok=True)

    rp.setup_style()
    conds, plans = [], []
    for d, label in zip(dirs, labels):
        runs, plan = rp.load(d)
        c = rp.aggregate(runs)
        c.insert(0, "exp", label)
        conds.append(c)
        plans.append(plan)
    cond = pd.concat(conds, ignore_index=True)
    cond.to_csv(out / "compare.csv", index=False)

    lines = [f"# 전후 비교: {' / '.join(labels)}", "",
             f"- 기준(최적화 전): **{labels[0]}**. 나머지는 같은 시나리오·같은 도착률에서 설정만 바꾼 실험입니다.",
             f"- goodput과 SLO 판정 기준: LLM TTFT p95 ≤ {rp.SLO['ttft_ms']:g} ms·TPOT p95 ≤ {rp.SLO['tpot_ms']:g} ms, "
             f"CV p95 ≤ {rp.SLO['cv_ms']:g} ms, 에러율 ≤ 1%. 도착률 옆 †는 측정이 무효였던 회차가 있는 단계입니다.", "",
             "## 실험 조건", "",
             rp.md_table(["실험", "시작", "커밋", "측정(초)", "반복"],
                         [[label, pl["started_at"], f"`{pl['git_commit']}`", f"{float(pl['args']['duration']):g}",
                           str(pl["args"]["repeat"])] for label, pl in zip(labels, plans)]), "",
             "### 바꾼 설정 (`.env`에서 실험마다 다른 값)", "", env_diff(labels, [read_env(d) for d in dirs]), "",
             "## 용량 요약", "",
             "SLO 용량 = 모든 요청을 받아들이면서(거절·에러 1% 이하) SLO를 지킨 가장 높은 도착률. "
             "최대 goodput = 측정한 도착률 중 SLO 안에 끝난 요청이 가장 많았던 값 (일부를 거절하는 설정은 이 값으로 비교).", "",
             rp.md_table(["시나리오", "실험", "SLO 용량 (req/s)", "최대 goodput (req/s)", "그때 도착률", "기준 워크로드"],
                         capacity_rows(cond, labels)), ""]
    figs = []
    for scenario in SPECS:
        if cond[cond["scenario"] == scenario]["exp"].nunique() < 2:
            continue
        lines += [f"## {scenario}", "", compare_table(cond, scenario, labels), ""]
        f = compare_fig(cond, scenario, labels, out)
        if f:
            figs.append(f)
            lines += [f"![{f.stem}]({f.name})", ""]
    (out / "compare.md").write_text("\n".join(lines))
    print(f"실험 {len(dirs)}개 비교 → {out}/")
    for f in [out / "compare.md", out / "compare.csv", *figs]:
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
