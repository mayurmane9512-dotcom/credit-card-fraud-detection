"""Aggregations behind the fraud-trend charts.

Both the static report (visualize.py, matplotlib) and the dashboard (app.py,
Plotly) use these functions, so the numbers always match.
"""
import pandas as pd

from src.preprocessing import DAY_NAMES, TARGET


def summary_stats(df):
    """Headline numbers for the dashboard KPI cards."""
    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]
    return {
        "transactions": len(df),
        "frauds": len(fraud),
        "fraud_rate_pct": 100 * len(fraud) / max(len(df), 1),
        "total_amount": df["amount"].sum(),
        "fraud_amount": fraud["amount"].sum(),
        "avg_fraud_amount": fraud["amount"].mean() if len(fraud) else 0.0,
        "avg_legit_amount": legit["amount"].mean() if len(legit) else 0.0,
        "customers": df["customer_id"].nunique() if "customer_id" in df else None,
    }


def fraud_rate_by(df, column):
    """Transactions, frauds and fraud rate (%) for each value of `column`."""
    out = (df.groupby(column, observed=True)[TARGET]
             .agg(transactions="count", frauds="sum")
             .reset_index())
    out["fraud_rate_pct"] = 100 * out["frauds"] / out["transactions"]
    return out


def daily_trend(df):
    """Fraud count, rate and amount per day, plus a 7-day moving average."""
    daily = (df.assign(fraud_amount=df["amount"].where(df[TARGET] == 1, 0.0))
               .groupby("date")
               .agg(transactions=(TARGET, "count"),
                    frauds=(TARGET, "sum"),
                    fraud_amount=("fraud_amount", "sum"))
               .reset_index())
    daily["fraud_rate_pct"] = 100 * daily["frauds"] / daily["transactions"]
    daily["frauds_7d_avg"] = daily["frauds"].rolling(7, min_periods=1).mean()
    return daily


def weekly_fraud_amount(df):
    """Money lost to fraud per week."""
    fraud = df[df[TARGET] == 1]
    weekly = (fraud.set_index("date")["amount"]
                   .resample("W-MON", label="left", closed="left").sum()
                   .reset_index())
    weekly.columns = ["week_start", "fraud_amount"]
    return weekly


def hour_weekday_matrix(df):
    """Fraud count for every (day of week, hour) pair - for a heatmap."""
    fraud = df[df[TARGET] == 1]
    matrix = pd.crosstab(fraud["day_name"], fraud["hour"])
    return matrix.reindex(index=DAY_NAMES, columns=range(24), fill_value=0)


def fraud_type_summary(df):
    """Count, total and average amount for each fraud type (modus operandi)."""
    fraud = df[df[TARGET] == 1]
    return (fraud.groupby("fraud_type")["amount"]
                 .agg(frauds="count", total_amount="sum", avg_amount="mean")
                 .reset_index()
                 .sort_values("frauds", ascending=False, ignore_index=True))


def risk_flag_rates(df):
    """Fraud rate with vs without each yes/no risk factor."""
    flags = {
        "is_online": ("Online", "In-store"),
        "is_international": ("International", "Domestic"),
        "is_new_merchant": ("New merchant", "Known merchant"),
        "is_night": ("Night (0-5h)", "Day"),
    }
    rows = []
    for col, (yes_label, no_label) in flags.items():
        for value, label in ((1, yes_label), (0, no_label)):
            subset = df[df[col] == value]
            rows.append({
                "factor": col,
                "group": label,
                "is_risky": value == 1,
                "transactions": len(subset),
                "fraud_rate_pct": 100 * subset[TARGET].mean() if len(subset) else 0.0,
            })
    return pd.DataFrame(rows)
