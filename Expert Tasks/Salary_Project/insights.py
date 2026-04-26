"""
src/insights.py
───────────────
Generates REAL, data-driven business insights from:
  - Actual dataset statistics (grouped means, medians)
  - Trained model's feature importances
  - Correlations between features and salary

PRINCIPLE: Every number shown comes from the actual data.
           No hard-coded percentages. No invented findings.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils import (
    get_logger, ALL_FEATURES, TARGET_COL,
    PIPELINE_PATH, FEAT_IMP_PATH, OUTPUTS_DIR,
    ORDINAL_FEATURES,
)

log = get_logger("insights")


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE (from saved CSV — generated during training)
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_importance(top_n: int = 20) -> pd.DataFrame | None:
    if not FEAT_IMP_PATH.exists():
        log.warning(f"Feature importance file not found: {FEAT_IMP_PATH}")
        return None
    fi = pd.read_csv(FEAT_IMP_PATH).head(top_n)
    return fi


# ─────────────────────────────────────────────────────────────────────────────
#  GROUPED SALARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def salary_by_group(df: pd.DataFrame, group_col: str, top_n: int = 15) -> pd.DataFrame:
    """
    Compute mean, median, and count of salary grouped by a categorical column.
    Sorted by mean salary descending.
    """
    grouped = (
        df.groupby(group_col)[TARGET_COL]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(top_n)
    )
    grouped.columns = [group_col, "avg_salary", "median_salary", "count"]
    return grouped


def salary_by_experience(df: pd.DataFrame) -> pd.DataFrame:
    """Avg salary by experience year (for trend line chart)."""
    return (
        df.groupby("experience_years")[TARGET_COL]
        .agg(avg_salary="mean", count="count")
        .reset_index()
        .sort_values("experience_years")
    )


def salary_by_education(df: pd.DataFrame) -> pd.DataFrame:
    """Avg salary by education level (ordered)."""
    order = ORDINAL_FEATURES["education_level"]
    grouped = df.groupby("education_level")[TARGET_COL].mean().reset_index()
    grouped.columns = ["education_level", "avg_salary"]
    # Sort in ordinal order, not alphabetical
    grouped["_ord"] = grouped["education_level"].apply(
        lambda x: order.index(x) if x in order else -1
    )
    return grouped.sort_values("_ord").drop(columns="_ord")


def salary_by_remote(df: pd.DataFrame) -> pd.DataFrame:
    """Avg salary by remote work status."""
    return (
        df.groupby("remote_work")[TARGET_COL]
        .agg(avg_salary="mean", count="count")
        .reset_index()
    )


def salary_by_certifications(df: pd.DataFrame) -> pd.DataFrame:
    """Avg salary by certification status."""
    return (
        df.groupby("certifications")[TARGET_COL]
        .agg(avg_salary="mean", count="count")
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATA-DRIVEN BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> list[dict]:
    """
    Compute real business findings from actual data.

    Returns a list of insight dicts, each with:
      - title    : short label
      - finding  : quantified statement from data
      - action   : business recommendation
      - value    : key number (for display)
    """
    insights = []

    # ── 1. Education premium ──────────────────────────────────────────────────
    edu_df = salary_by_education(df)
    if len(edu_df) >= 2:
        top_edu  = edu_df.iloc[-1]["avg_salary"]    # highest education
        base_edu = edu_df.iloc[0]["avg_salary"]     # lowest education
        pct_diff = (top_edu - base_edu) / base_edu * 100
        top_name = edu_df.iloc[-1]["education_level"]
        bot_name = edu_df.iloc[0]["education_level"]
        insights.append({
            "icon":    "🎓",
            "title":   "Education Premium",
            "finding": (
                f"{top_name} holders earn on average ${top_edu:,.0f}/yr "
                f"vs ${base_edu:,.0f}/yr for {bot_name} — a {pct_diff:.0f}% gap."
            ),
            "action":  "Offer tuition reimbursement programs to retain talent cheaper than external hiring.",
            "value":   f"+{pct_diff:.0f}%",
        })

    # ── 2. Experience salary growth ───────────────────────────────────────────
    exp_df = salary_by_experience(df)
    if len(exp_df) >= 5:
        young = exp_df[exp_df["experience_years"] <= 3]["avg_salary"].mean()
        mid   = exp_df[(exp_df["experience_years"] >= 4) & (exp_df["experience_years"] <= 8)]["avg_salary"].mean()
        pct   = (mid - young) / young * 100
        insights.append({
            "icon":    "📈",
            "title":   "Experience Growth Curve",
            "finding": (
                f"Salary jumps {pct:.0f}% from 0–3 yrs (avg ${young:,.0f}) "
                f"to 4–8 yrs (avg ${mid:,.0f})."
            ),
            "action":  "Implement structured 3-year and 5-year salary reviews to retain rising talent.",
            "value":   f"+{pct:.0f}% (yrs 4–8)",
        })

    # ── 3. Remote work premium ────────────────────────────────────────────────
    remote_df = salary_by_remote(df)
    if len(remote_df) >= 2:
        remote_sal = remote_df[remote_df["remote_work"].str.lower() == "yes"]["avg_salary"]
        office_sal = remote_df[remote_df["remote_work"].str.lower() == "no"]["avg_salary"]
        if len(remote_sal) and len(office_sal):
            diff     = remote_sal.values[0] - office_sal.values[0]
            diff_pct = diff / office_sal.values[0] * 100
            direction = "higher" if diff > 0 else "lower"
            insights.append({
                "icon":    "🏠",
                "title":   "Remote Work Impact",
                "finding": (
                    f"Remote roles pay ${abs(diff):,.0f} ({abs(diff_pct):.1f}%) {direction} "
                    f"on average than on-site positions."
                ),
                "action":  "Formalise a remote-work pay policy — it affects talent attraction and budget planning.",
                "value":   f"{'+' if diff>0 else ''}{diff_pct:.1f}% vs in-office",
            })

    # ── 4. Certification premium ──────────────────────────────────────────────
    cert_df = salary_by_certifications(df)
    if len(cert_df) >= 2:
        with_cert = cert_df[
        cert_df["certifications"].astype(str).str.lower().isin(["yes", "1", "true"])
        ]["avg_salary"]
        without_cert = cert_df[
        cert_df["certifications"].astype(str).str.lower().isin(["no", "0", "false"])
        ]["avg_salary"]
        if len(with_cert) and len(without_cert):
            pct = (with_cert.values[0] - without_cert.values[0]) / without_cert.values[0] * 100
            insights.append({
                "icon":    "📜",
                "title":   "Certification Premium",
                "finding": (
                    f"Certified candidates earn {pct:.1f}% more on average "
                    f"(${with_cert.values[0]:,.0f} vs ${without_cert.values[0]:,.0f})."
                ),
                "action":  "Reimburse professional certifications — faster ROI than salary-driven replacement hiring.",
                "value":   f"+{pct:.1f}% avg salary",
            })

    # ── 5. Top-paying industry ────────────────────────────────────────────────
    ind_df = salary_by_group(df, "industry", top_n=3)
    if len(ind_df) >= 2:
        top_ind  = ind_df.iloc[0]
        last_ind = ind_df.iloc[-1] if len(ind_df) > 2 else None
        finding  = (
            f"'{top_ind['industry']}' pays the highest average salary at "
            f"${top_ind['avg_salary']:,.0f}/yr."
        )
        if last_ind is not None:
            gap = top_ind["avg_salary"] - last_ind["avg_salary"]
            finding += f" That's ${gap:,.0f} more than '{last_ind['industry']}'."
        insights.append({
            "icon":    "🏭",
            "title":   "Industry Pay Gap",
            "finding": finding,
            "action":  "Benchmark salaries against top-paying industries quarterly to stay competitive.",
            "value":   f"${top_ind['avg_salary']:,.0f} avg",
        })

    # ── 6. Skills multiplier ──────────────────────────────────────────────────
    low_skills  = df[df["skills_count"] <= 4][TARGET_COL].mean()
    high_skills = df[df["skills_count"] >= 10][TARGET_COL].mean()
    if low_skills > 0:
        pct_skills = (high_skills - low_skills) / low_skills * 100
        insights.append({
            "icon":    "🛠️",
            "title":   "Skills Count Multiplier",
            "finding": (
                f"Profiles with 10+ skills earn ${high_skills:,.0f}/yr vs "
                f"${low_skills:,.0f}/yr for ≤4 skills — a {pct_skills:.0f}% difference."
            ),
            "action":  "Invest in internal upskilling — cheaper than hiring externally for each new skill set.",
            "value":   f"+{pct_skills:.0f}% (10+ vs ≤4 skills)",
        })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
#  PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_insight_report(insights: list[dict]) -> None:
    print("\n" + "═" * 65)
    print("  BUSINESS INSIGHTS  (data-driven, from actual dataset)")
    print("═" * 65)
    for i, ins in enumerate(insights, 1):
        print(f"\n  {ins['icon']}  [{i}] {ins['title']}  →  {ins['value']}")
        print(f"      Finding : {ins['finding']}")
        print(f"      Action  : {ins['action']}")
    print("\n" + "═" * 65)


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/salary_data.csv"
    import pandas as pd
    df = pd.read_csv(data_path)
    insights = generate_insights(df)
    print_insight_report(insights)
