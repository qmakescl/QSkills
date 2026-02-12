#!/usr/bin/env python3
"""
DataFrame 기초 통계 산출 스크립트
=================================
CLI 사용:
    python compute_stats.py datasets/titanic.csv
    python compute_stats.py datasets/titanic.csv --no-html
    python compute_stats.py datasets/titanic.csv --no-md

모듈 사용:
    from compute_stats import compute_basic_stats
    result = compute_basic_stats(df)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 타입 추론
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def infer_column_type(series: pd.Series) -> str:
    """
    열의 dtype을 분석하여 4가지 범주 중 하나를 반환.
      continuous | integer | datetime | categorical
    """
    dtype = series.dtype

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    if pd.api.types.is_integer_dtype(dtype):
        return "integer"

    if pd.api.types.is_float_dtype(dtype):
        return "continuous"

    # object 열 → 내부 값 기반 2차 추론
    if dtype == object:
        non_null = series.dropna()
        if len(non_null) == 0:
            return "categorical"

        # 숫자 변환 시도
        converted = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = converted.notna().sum() / len(non_null)
        if numeric_ratio > 0.8:
            if (converted.dropna() % 1 == 0).all():
                return "integer"
            return "continuous"

        # 날짜 변환 시도
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() / len(non_null) > 0.8:
                return "datetime"
        except Exception:
            pass

    return "categorical"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통계 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_round(val, decimals: int = 4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


def _compute_common(series: pd.Series) -> dict[str, Any]:
    """모든 타입에 공통되는 기본 통계."""
    total = len(series)
    missing = int(series.isna().sum())
    return {
        "column_name": series.name,
        "total_count": total,
        "valid_count": total - missing,
        "missing_count": missing,
        "missing_rate": round(missing / total, 4) if total else None,
    }


def _compute_numeric(series: pd.Series) -> dict[str, Any]:
    """연속형·정수형 공통 수치 통계."""
    s = pd.to_numeric(series, errors="coerce")
    return {
        "mean": _safe_round(s.mean()),
        "std": _safe_round(s.std()),
        "median": _safe_round(s.median()),
        "min": _safe_round(s.min()),
        "max": _safe_round(s.max()),
    }


def _compute_mode(series: pd.Series) -> Any:
    """최빈값 계산 (빈 시리즈 안전 처리)."""
    mode_vals = series.mode()
    if len(mode_vals) == 0:
        return None
    val = mode_vals.iloc[0]
    # numpy/pandas 타입 → Python 네이티브 변환
    if hasattr(val, "item"):
        return val.item()
    return val


MAX_CATEGORY_FOR_DISTRIBUTION = 8


def _generate_categorical_chart(
    series: pd.Series,
    col_name: str,
    output_dir: str | None = None,
) -> tuple[str | None, str | None]:
    """
    범주형 변수(고유값 8개 이하)의 가로 막대 차트를 생성한다.

    Returns
    -------
    (png_path | None, base64_string | None)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None

    value_counts = series.value_counts()
    total = value_counts.sum()

    n = len(value_counts)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.6)))
    bars = ax.barh(
        [str(v) for v in value_counts.index],
        value_counts.values,
        color="#4C78A8",
    )
    for bar, count in zip(bars, value_counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Count")
    ax.set_title(f"{col_name} Distribution")
    ax.invert_yaxis()
    x_margin = total * 0.15
    ax.set_xlim(0, value_counts.values.max() + x_margin)
    plt.tight_layout()

    # base64 인코딩
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    buf.close()

    # PNG 파일 저장
    png_path = None
    if output_dir:
        charts_dir = Path(output_dir) / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        png_path = str(charts_dir / f"{col_name}_distribution.png")
        fig.savefig(png_path, dpi=100, bbox_inches="tight")

    plt.close(fig)
    return png_path, b64


def _build_distribution_description(series: pd.Series) -> str:
    """범주형 변수의 분포를 자연어로 요약한다."""
    value_counts = series.value_counts()
    total = value_counts.sum()
    unique = len(value_counts)
    parts = []
    for val, cnt in value_counts.items():
        pct = cnt / total * 100
        parts.append(f"'{val}': {cnt:,}건({pct:.1f}%)")
    return f"총 {unique}개 범주. " + ", ".join(parts)


def stats_for_column(
    series: pd.Series,
    col_type: str,
    chart_output_dir: str | None = None,
) -> dict[str, Any]:
    """열 하나에 대해 타입별 맞춤 통계를 산출."""
    stats = _compute_common(series)
    stats["inferred_type"] = col_type

    if col_type == "continuous":
        stats.update(_compute_numeric(series))

    elif col_type == "integer":
        stats.update(_compute_numeric(series))
        stats["mode"] = _compute_mode(pd.to_numeric(series, errors="coerce"))

    elif col_type == "datetime":
        s = pd.to_datetime(series, errors="coerce")
        stats["min"] = str(s.min()) if pd.notna(s.min()) else None
        stats["max"] = str(s.max()) if pd.notna(s.max()) else None
        stats["mode"] = str(_compute_mode(s)) if _compute_mode(s) is not None else None

    elif col_type == "categorical":
        value_counts = series.value_counts()
        unique_count = int(series.nunique())
        stats["unique_count"] = unique_count
        stats["mode"] = _compute_mode(series)

        if unique_count <= MAX_CATEGORY_FOR_DISTRIBUTION:
            # 전체 범주의 분포 (건수 + 비율)
            total = int(value_counts.sum())
            stats["distribution"] = {
                str(k): {"count": int(v), "ratio": round(v / total, 4)}
                for k, v in value_counts.items()
            }
            stats["distribution_description"] = _build_distribution_description(series)
            png_path, b64 = _generate_categorical_chart(
                series, series.name, output_dir=chart_output_dir
            )
            stats["distribution_chart"] = png_path
            stats["distribution_chart_base64"] = b64
        else:
            stats["top5_values"] = {
                str(k): int(v) for k, v in value_counts.head(5).items()
            }

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ydata-profiling 리포트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_profile_report(
    df: pd.DataFrame,
    output_path: str,
    title: str = "DataFrame Profile Report",
    minimal: bool = True,
) -> str | None:
    """ydata-profiling HTML 리포트 생성. 미설치 시 None 반환."""
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        warnings.warn(
            "[df-basic-stats] ydata-profiling 미설치. "
            "프로파일 리포트를 건너뜁니다. "
            "설치: pip install ydata-profiling",
            stacklevel=2,
        )
        return None

    # 대용량 데이터 시 minimal 강제
    if len(df) > 1_000_000:
        minimal = True

    profile = ProfileReport(df, title=title, minimal=minimal, explorative=False)
    profile.to_file(output_path)
    return output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Markdown 리포트 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _generate_insights(result: dict) -> list[str]:
    """통계 결과에서 자동 인사이트를 추출한다."""
    insights = []
    rows = result["dataframe_shape"]["rows"]

    for col in result["columns"]:
        name = col["column_name"]
        mr = col.get("missing_rate") or 0
        ctype = col["inferred_type"]

        # 높은 결측률
        if mr >= 0.5:
            insights.append(
                f"**{name}** 변수의 결측률이 {mr*100:.1f}%로 매우 높아 "
                "변수 삭제 또는 유무(binary) 변환을 고려할 필요가 있다."
            )
        elif mr >= 0.1:
            insights.append(
                f"**{name}** 변수의 결측률이 {mr*100:.1f}%로 "
                "결측 처리(imputation) 전략이 필요하다."
            )

        # 분포 치우침 (수치형)
        if ctype in ("continuous", "integer"):
            mean = col.get("mean")
            median = col.get("median")
            if mean and median and median != 0:
                ratio = mean / median
                if ratio > 1.5:
                    insights.append(
                        f"**{name}** 변수의 평균({mean})이 중앙값({median})의 "
                        f"{ratio:.1f}배로 우측 꼬리가 긴(right-skewed) 분포를 보인다."
                    )
                elif ratio < 0.67:
                    insights.append(
                        f"**{name}** 변수의 평균({mean})이 중앙값({median})보다 "
                        "현저히 낮아 좌측 꼬리가 긴(left-skewed) 분포를 보인다."
                    )

        # 범주형 인사이트
        if ctype == "categorical":
            uc = col.get("unique_count", 0)
            dist = col.get("distribution")

            if uc == rows:
                insights.append(
                    f"**{name}** 변수는 모든 값이 고유하여 식별자(identifier) 역할을 한다."
                )
            elif uc == 2:
                insights.append(f"**{name}** 변수는 이진(binary) 변수이다.")

            if dist:
                top_val = next(iter(dist))
                top_ratio = dist[top_val]["ratio"]
                if top_ratio >= 0.7:
                    insights.append(
                        f"**{name}** 변수에서 '{top_val}'이(가) "
                        f"{top_ratio*100:.1f}%로 지배적이다."
                    )

        # 이진 결과 변수 (integer, 0/1)
        if ctype == "integer":
            if col.get("min") == 0 and col.get("max") == 1:
                mean_val = col.get("mean")
                if mean_val:
                    insights.append(
                        f"**{name}** 변수는 이진 결과 변수로, "
                        f"양성(1) 비율이 {mean_val*100:.1f}%이다."
                    )

    return insights


def generate_stats_markdown(
    result: dict,
    dataset_name: str,
    output_dir: str,
    profile_html_path: str | None = None,
) -> str:
    """통계 결과를 Markdown 리포트 문자열로 생성한다."""
    rows = result["dataframe_shape"]["rows"]
    cols = result["dataframe_shape"]["cols"]

    # 전체 결측 계산
    total_missing = sum(c.get("missing_count", 0) for c in result["columns"])
    total_cells = rows * cols
    total_missing_rate = total_missing / total_cells * 100 if total_cells else 0

    lines: list[str] = []
    lines.append(f"# {dataset_name} 기초 통계 분석 리포트\n")

    # ── 1. 데이터 개요 ──
    lines.append("## 1. 데이터 개요\n")
    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 행 수 | {rows:,} |")
    lines.append(f"| 열 수 | {cols} |")
    lines.append(f"| 전체 결측 셀 | {total_missing:,} |")
    lines.append(f"| 전체 결측률 | {total_missing_rate:.1f}% |")
    lines.append("")

    # ── 2. 결측값 현황 ──
    lines.append("## 2. 결측값 현황\n")
    missing_cols = [
        c for c in result["columns"] if (c.get("missing_count") or 0) > 0
    ]
    no_missing_count = len(result["columns"]) - len(missing_cols)

    if missing_cols:
        missing_cols.sort(key=lambda c: c["missing_count"], reverse=True)
        lines.append("| 변수 | 결측수 | 결측률 | 상태 |")
        lines.append("|------|--------|--------|------|")
        for c in missing_cols:
            mc = c["missing_count"]
            mr = (c.get("missing_rate") or 0) * 100
            if mr >= 10:
                status = "⚠️ 높음" if mr >= 50 else "⚠️ 주의"
            else:
                status = "✅"
            lines.append(f"| {c['column_name']} | {mc:,} | {mr:.1f}% | {status} |")
        if no_missing_count > 0:
            lines.append(f"\n나머지 {no_missing_count}개 변수는 결측 없음.")
    else:
        lines.append("모든 변수에 결측값이 없습니다.")
    lines.append("")

    # ── 3. 변수별 통계 ──
    lines.append("## 3. 변수별 통계\n")

    numeric_cols = [
        c for c in result["columns"]
        if c["inferred_type"] in ("continuous", "integer")
    ]
    cat_cols = [
        c for c in result["columns"]
        if c["inferred_type"] == "categorical"
    ]
    dt_cols = [
        c for c in result["columns"]
        if c["inferred_type"] == "datetime"
    ]

    # 3.1 수치형
    if numeric_cols:
        lines.append("### 3.1 수치형 변수 (continuous / integer)\n")
        lines.append("| 변수 | 타입 | 평균 | 표준편차 | 중앙값 | 최솟값 | 최댓값 | 최빈값 |")
        lines.append("|------|------|------|----------|--------|--------|--------|--------|")
        for c in numeric_cols:
            def _fmt(v):
                if v is None:
                    return "-"
                return f"{v:,.4f}".rstrip("0").rstrip(".")
            mode_val = c.get("mode")
            mode_str = str(mode_val) if mode_val is not None else "-"
            lines.append(
                f"| {c['column_name']} | {c['inferred_type']} "
                f"| {_fmt(c.get('mean'))} | {_fmt(c.get('std'))} "
                f"| {_fmt(c.get('median'))} | {_fmt(c.get('min'))} "
                f"| {_fmt(c.get('max'))} | {mode_str} |"
            )
        lines.append("")

    # 3.2 범주형
    if cat_cols:
        lines.append("### 3.2 범주형 변수 (categorical)\n")
        for c in cat_cols:
            uc = c.get("unique_count", 0)
            lines.append(f"#### {c['column_name']} ({uc}개 범주)\n")

            dist = c.get("distribution")
            if dist:
                lines.append("| 범주 | 건수 | 비율 |")
                lines.append("|------|------|------|")
                for val, info in dist.items():
                    lines.append(
                        f"| {val} | {info['count']:,} | {info['ratio']*100:.1f}% |"
                    )
                lines.append("")
                # 차트 이미지 링크
                chart_path = c.get("distribution_chart")
                if chart_path:
                    rel_chart = f"../charts/{c['column_name']}_distribution.png"
                    lines.append(f"![{c['column_name']} 분포]({rel_chart})\n")
            else:
                top5 = c.get("top5_values")
                if top5 and uc > MAX_CATEGORY_FOR_DISTRIBUTION:
                    lines.append(
                        f"> 고유값이 {MAX_CATEGORY_FOR_DISTRIBUTION}개를 초과하여 상위 5개만 표시\n"
                    )
                    lines.append("| 값 | 건수 |")
                    lines.append("|-----|------|")
                    for val, cnt in top5.items():
                        lines.append(f"| {val} | {cnt:,} |")
                    lines.append("")

    # 3.3 날짜형
    if dt_cols:
        lines.append("### 3.3 날짜형 변수 (datetime)\n")
        lines.append("| 변수 | 최솟값 | 최댓값 | 최빈값 |")
        lines.append("|------|--------|--------|--------|")
        for c in dt_cols:
            lines.append(
                f"| {c['column_name']} | {c.get('min', '-')} "
                f"| {c.get('max', '-')} | {c.get('mode', '-')} |"
            )
        lines.append("")

    # ── 4. 주요 인사이트 ──
    insights = _generate_insights(result)
    lines.append("## 4. 주요 인사이트\n")
    if insights:
        for ins in insights:
            lines.append(f"- {ins}")
    else:
        lines.append("- 특이사항 없음.")
    lines.append("")

    # ── 5. 참고 ──
    lines.append("## 5. 참고\n")
    if profile_html_path:
        html_name = Path(profile_html_path).name
        lines.append(f"- 상세 프로파일 리포트: [{html_name}](../{html_name})")
    else:
        lines.append("- 상세 프로파일 리포트: 생성되지 않음 (--no-html)")
    lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 함수 (모듈 & CLI 공용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_basic_stats(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    profile_report_path: str | None = "report.html",
    profile_minimal: bool = True,
    chart_output_dir: str | None = None,
    md_output_path: str | None = None,
) -> dict[str, Any]:
    """
    데이터프레임의 기초 통계를 산출한다.

    Parameters
    ----------
    df : pd.DataFrame
    dataset_name : str
        데이터셋 이름 (MD 리포트 제목에 사용).
    profile_report_path : str | None
        HTML 리포트 저장 경로. None이면 리포트 생성 안 함.
    profile_minimal : bool
        True면 경량 모드.
    chart_output_dir : str | None
        범주형 분포 차트 PNG 저장 디렉토리. None이면 PNG 저장을 건너뛴다.
        base64와 description은 항상 생성된다.
    md_output_path : str | None
        Markdown 리포트 저장 경로. None이면 MD 생성 안 함.

    Returns
    -------
    dict  (구조는 SKILL.md '출력 형식' 참조)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"pd.DataFrame 필요 (받은 타입: {type(df).__name__})")

    if len(df) == 0:
        return {
            "dataframe_shape": {"rows": 0, "cols": len(df.columns)},
            "columns": [],
            "profile_report": None,
            "_warning": "빈 데이터프레임입니다.",
        }

    # 열별 타입 추론 + 통계 산출
    columns_stats = []
    for col in df.columns:
        col_type = infer_column_type(df[col])
        col_stats = stats_for_column(df[col], col_type, chart_output_dir=chart_output_dir)
        columns_stats.append(col_stats)

    # 프로파일 리포트
    report = None
    if profile_report_path:
        report = generate_profile_report(
            df, profile_report_path, minimal=profile_minimal
        )

    result = {
        "dataframe_shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": columns_stats,
        "profile_report": report,
    }

    # Markdown 리포트 생성
    if md_output_path:
        md_dir = Path(md_output_path).parent
        md_dir.mkdir(parents=True, exist_ok=True)
        md_content = generate_stats_markdown(
            result, dataset_name, str(md_dir),
            profile_html_path=profile_report_path,
        )
        Path(md_output_path).write_text(md_content, encoding="utf-8")
        result["markdown_report"] = md_output_path

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일 로더 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataframe(file_path: str) -> pd.DataFrame:
    """확장자 기반으로 적절한 pandas reader를 선택하여 로드."""
    p = Path(file_path)
    ext = p.suffix.lower()

    loaders = {
        ".csv": pd.read_csv,
        ".tsv": lambda f: pd.read_csv(f, sep="\t"),
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".feather": pd.read_feather,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"지원하지 않는 파일 형식: {ext}\n"
            f"지원 형식: {', '.join(loaders.keys())}"
        )
    return loader(file_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI 엔트리포인트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="DataFrame 기초 통계 산출 (Agent Skill)"
    )
    parser.add_argument("input_file", help="분석할 데이터 파일 경로")
    parser.add_argument(
        "--output-json", default=None,
        help="통계 결과 JSON 저장 경로 (미지정 시 자동: {dataset_name}-stats.json)"
    )
    parser.add_argument(
        "--output-html", default=None,
        help="ydata-profiling HTML 리포트 저장 경로 (미지정 시 자동: {dataset_name}-ydata-profiling.html)"
    )
    parser.add_argument(
        "--output-md", default=None,
        help="Markdown 리포트 저장 경로 (미지정 시 자동: {dataset_name}-stats.md)"
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="HTML 리포트 생성을 건너뛴다"
    )
    parser.add_argument(
        "--no-md", action="store_true",
        help="Markdown 리포트 생성을 건너뛴다"
    )
    parser.add_argument(
        "--full-profile", action="store_true",
        help="전체(full) 프로파일링 모드 (기본: minimal)"
    )
    args = parser.parse_args()

    # 데이터셋 이름 및 출력 디렉토리
    dataset_name = Path(args.input_file).stem
    output_dir = str(Path(args.input_file).parent)

    # 자동 파일명 결정
    html_path = None if args.no_html else (
        args.output_html or f"{output_dir}/{dataset_name}-ydata-profiling.html"
    )
    json_path = args.output_json or f"{output_dir}/{dataset_name}-stats.json"
    md_path = None if args.no_md else (
        args.output_md or f"{output_dir}/report/{dataset_name}-stats.md"
    )

    # 차트 출력 디렉토리 (입력 파일과 같은 디렉토리)
    chart_dir = output_dir

    # 로드
    df = load_dataframe(args.input_file)
    print(f"[INFO] 로드 완료: {df.shape[0]}행 × {df.shape[1]}열", file=sys.stderr)

    # 분석
    result = compute_basic_stats(
        df,
        dataset_name=dataset_name,
        profile_report_path=html_path,
        profile_minimal=not args.full_profile,
        chart_output_dir=chart_dir,
        md_output_path=md_path,
    )

    # JSON 출력
    output_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    Path(json_path).write_text(output_json, encoding="utf-8")
    print(f"[INFO] JSON 저장: {json_path}", file=sys.stderr)

    if result.get("profile_report"):
        print(f"[INFO] 프로파일 리포트: {result['profile_report']}", file=sys.stderr)

    if result.get("markdown_report"):
        print(f"[INFO] Markdown 리포트: {result['markdown_report']}", file=sys.stderr)


if __name__ == "__main__":
    main()
