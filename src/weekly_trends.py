from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out.columns = [c.strip().lower() for c in out.columns]

    required = {"event_date", "department", "visits", "revenue"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out["department"] = out["department"].astype("object").str.strip().str.title()

    tmp = (
        out["event_date"]
        .astype("object")
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )
    out["event_date"] = pd.to_datetime(tmp, errors="coerce")
    out = out.dropna(subset=["event_date"])

    out["visits"] = pd.to_numeric(out["visits"], errors="coerce")
    out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce")

    return out


def weekly_aggregate_overall(df: pd.DataFrame) -> pd.DataFrame:
    wk = (
        df.set_index("event_date")
          .resample("W-SUN")[["visits", "revenue"]]
          .sum(min_count=1)
          .reset_index()
          .rename(columns={"event_date": "week_end"})
    )

    wk["visits_wow_pct"] = wk["visits"].pct_change() * 100
    wk["revenue_wow_pct"] = wk["revenue"].pct_change() * 100

    return wk


def weekly_aggregate_by_department(df: pd.DataFrame) -> pd.DataFrame:
    wk_dept = (
        df.assign(week_end=df["event_date"].dt.to_period("W-SUN").dt.end_time.dt.normalize())
          .groupby(["week_end", "department"], as_index=False)[["visits", "revenue"]]
          .sum(min_count=1)
          .sort_values(["department", "week_end"])
    )

    wk_dept["visits_wow_pct"] = wk_dept.groupby("department")["visits"].pct_change() * 100
    wk_dept["revenue_wow_pct"] = wk_dept.groupby("department")["revenue"].pct_change() * 100

    return wk_dept


def flag_changes(series_pct: pd.Series, threshold: float) -> pd.Series:
    flags = np.where(series_pct >= threshold, "UP",
             np.where(series_pct <= -threshold, "DOWN", ""))
    return pd.Series(flags, index=series_pct.index)


def generate_report(
    report_path: Path,
    raw_rows: int,
    cleaned_rows: int,
    overall: pd.DataFrame,
    by_dept: pd.DataFrame,
    threshold_pct: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("WEEKLY TREND REPORT")
    lines.append("=" * 28)
    lines.append("")
    lines.append("Data summary")
    lines.append(f"- Rows (raw): {raw_rows}")
    lines.append(f"- Rows (after date parse): {cleaned_rows}")
    lines.append(f"- Weeks analyzed: {int(overall.shape[0])}")
    lines.append(f"- Spike/Drop threshold: {threshold_pct:.0f}% week-over-week")
    lines.append("")

    lines.append("Overall weekly totals")
    disp = overall.copy()
    disp["visits_flag"] = flag_changes(disp["visits_wow_pct"], threshold_pct)
    disp["revenue_flag"] = flag_changes(disp["revenue_wow_pct"], threshold_pct)

    show_cols = ["week_end", "visits", "visits_wow_pct", "visits_flag", "revenue", "revenue_wow_pct", "revenue_flag"]
    lines.append(disp[show_cols].to_string(index=False))
    lines.append("")

    if overall.shape[0] > 1:
        lines.append("Largest week-over-week movers (overall)")
        top_visits_up = overall.loc[overall["visits_wow_pct"].idxmax()] if overall["visits_wow_pct"].notna().any() else None
        top_visits_down = overall.loc[overall["visits_wow_pct"].idxmin()] if overall["visits_wow_pct"].notna().any() else None
        top_rev_up = overall.loc[overall["revenue_wow_pct"].idxmax()] if overall["revenue_wow_pct"].notna().any() else None
        top_rev_down = overall.loc[overall["revenue_wow_pct"].idxmin()] if overall["revenue_wow_pct"].notna().any() else None

        def fmt_row(label: str, row: pd.Series, pct_col: str) -> str:
            if row is None or pd.isna(row[pct_col]):
                return f"- {label}: n/a"
            return f"- {label}: {row['week_end'].date()} ({row[pct_col]:.1f}%)"

        lines.append(fmt_row("Visits biggest increase", top_visits_up, "visits_wow_pct"))
        lines.append(fmt_row("Visits biggest decrease", top_visits_down, "visits_wow_pct"))
        lines.append(fmt_row("Revenue biggest increase", top_rev_up, "revenue_wow_pct"))
        lines.append(fmt_row("Revenue biggest decrease", top_rev_down, "revenue_wow_pct"))
        lines.append("")

    lines.append("Department trend flags (threshold-based)")
    by = by_dept.copy()
    by["visits_flag"] = flag_changes(by["visits_wow_pct"], threshold_pct)
    by["revenue_flag"] = flag_changes(by["revenue_wow_pct"], threshold_pct)

    flagged = by[(by["visits_flag"] != "") | (by["revenue_flag"] != "")]
    if flagged.empty:
        lines.append("(No department-level spikes/drops detected.)")
    else:
        lines.append(flagged[["week_end", "department", "visits_wow_pct", "visits_flag", "revenue_wow_pct", "revenue_flag"]]
                     .to_string(index=False))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate weekly trends and generate a report.")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to output weekly summary CSV")
    parser.add_argument("--report", default="reports/weekly_trend_report.txt", help="Path to report output")
    parser.add_argument("--threshold", type=float, default=20.0, help="WoW threshold percent for spike/drop flags")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    threshold_pct = float(args.threshold)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path)
    cleaned = clean_input(raw)

    overall = weekly_aggregate_overall(cleaned)
    by_dept = weekly_aggregate_by_department(cleaned)

    # Save output CSV (department-level summary is most useful for BI)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_dept.to_csv(output_path, index=False)

    generate_report(
        report_path=report_path,
        raw_rows=int(raw.shape[0]),
        cleaned_rows=int(cleaned.shape[0]),
        overall=overall,
        by_dept=by_dept,
        threshold_pct=threshold_pct,
    )

    print("✅ Done!")
    print(f"- Weekly summary CSV: {output_path}")
    print(f"- Report: {report_path}")


if __name__ == "__main__":
    main()
