"""Exploratory data analysis (EDA): fraud-trend and transaction charts.

Every chart is saved as a PNG in reports/figures/ so it can be pasted into
a project report or presentation.
"""
import matplotlib

matplotlib.use("Agg")  # draw to files only - no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import analysis, config
from src.preprocessing import DAY_NAMES, SKEWED_FEATURES, TARGET

LEGIT, FRAUD = config.COLOR_LEGIT, config.COLOR_FRAUD
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(df):
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Genuine", "Fraud"], counts.values, color=[LEGIT, FRAUD])
    for bar, count in zip(bars, counts.values):
        ax.annotate(f"{count:,}\n({count / len(df):.2%})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Transactions")
    ax.set_title("Class distribution - fraud is rare (imbalanced data)")
    ax.set_ylim(0, counts.max() * 1.18)
    return _save(fig, "01_class_distribution.png")


def plot_daily_trend(df):
    daily = analysis.daily_trend(df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax1.bar(daily["date"], daily["frauds"], color=FRAUD, alpha=0.35, label="Frauds per day")
    ax1.plot(daily["date"], daily["frauds_7d_avg"], color=FRAUD, lw=2.2, label="7-day average")
    ax1.set_ylabel("Fraud transactions")
    ax1.set_title("Daily fraud trend")
    ax1.legend(loc="upper left")
    ax2.plot(daily["date"], daily["fraud_rate_pct"], color="#6C4AB6", lw=1.6)
    ax2.set_ylabel("Fraud rate (%)")
    ax2.set_xlabel("Date")
    fig.autofmt_xdate()
    return _save(fig, "02_daily_fraud_trend.png")


def plot_fraud_by_hour(df):
    by_hour = analysis.fraud_rate_by(df, "hour").set_index("hour").reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(by_hour.index, by_hour["transactions"], color=LEGIT, alpha=0.3, label="All transactions")
    ax.set_ylabel("Transactions")
    ax.set_xlabel("Hour of day")
    ax.set_xticks(range(24))
    ax2 = ax.twinx()
    ax2.plot(by_hour.index, by_hour["fraud_rate_pct"], color=FRAUD, marker="o", lw=2, label="Fraud rate")
    ax2.set_ylabel("Fraud rate (%)", color=FRAUD)
    ax2.grid(False)
    # headroom on both axes so the legend does not cover the bars or the line
    ax.set_ylim(0, by_hour["transactions"].max() * 1.25)
    ax2.set_ylim(0, by_hour["fraud_rate_pct"].max() * 1.25)
    ax.set_title("Fraud rate by hour - fraudsters strike at night while volume is low")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, ["All transactions", "Fraud rate"], loc="upper center")
    return _save(fig, "03_fraud_by_hour.png")


def plot_fraud_by_category(df):
    by_cat = analysis.fraud_rate_by(df, "merchant_category").sort_values("fraud_rate_pct")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [FRAUD if r > df[TARGET].mean() * 100 else LEGIT for r in by_cat["fraud_rate_pct"]]
    ax.barh(by_cat["merchant_category"], by_cat["fraud_rate_pct"], color=colors)
    ax.axvline(df[TARGET].mean() * 100, color="grey", ls="--", lw=1, label="Overall fraud rate")
    for y, (rate, frauds) in enumerate(zip(by_cat["fraud_rate_pct"], by_cat["frauds"])):
        ax.text(rate, y, f"  {rate:.1f}%  ({frauds})", va="center", fontsize=9)
    ax.set_xlabel("Fraud rate (%)  (number of frauds in brackets)")
    ax.set_title("Fraud rate by merchant category")
    ax.legend(loc="lower right")
    ax.set_xlim(0, by_cat["fraud_rate_pct"].max() * 1.3)
    return _save(fig, "04_fraud_by_category.png")


def plot_amount_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.logspace(0, np.log10(df["amount"].max() + 1), 50)
    for cls, color, label in ((0, LEGIT, "Genuine"), (1, FRAUD, "Fraud")):
        amounts = df.loc[df[TARGET] == cls, "amount"]
        # weights -> each class sums to 100%, so the rare fraud class is visible
        ax1.hist(amounts, bins=bins, color=color, alpha=0.6, label=label,
                 weights=np.full(len(amounts), 100 / max(len(amounts), 1)))
    ax1.set_xscale("log")
    ax1.set_xlabel("Amount (log scale)")
    ax1.set_ylabel("% of the class's transactions")
    ax1.set_title("Amount distribution")
    ax1.legend()
    box = ax2.boxplot([df.loc[df[TARGET] == 0, "amount_to_avg_ratio"],
                       df.loc[df[TARGET] == 1, "amount_to_avg_ratio"]],
                      tick_labels=["Genuine", "Fraud"], showfliers=False, patch_artist=True,
                      widths=0.5, medianprops={"color": "black"})
    for patch, color in zip(box["boxes"], [LEGIT, FRAUD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("Amount / customer's usual amount (log)")
    ax2.set_title("Fraud amounts are unusual for the customer")
    return _save(fig, "05_amount_distribution.png")


def plot_fraud_by_weekday(df):
    by_day = analysis.fraud_rate_by(df, "day_name").set_index("day_name").reindex(DAY_NAMES)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [FRAUD if d in ("Sat", "Sun") else LEGIT for d in DAY_NAMES]
    ax.bar(DAY_NAMES, by_day["fraud_rate_pct"], color=colors)
    for x, rate in enumerate(by_day["fraud_rate_pct"]):
        ax.text(x, rate, f"{rate:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by day of week (weekends highlighted)")
    return _save(fig, "06_fraud_by_weekday.png")


def plot_risk_factors(df):
    rates = analysis.risk_flag_rates(df)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, (factor, group) in zip(axes, rates.groupby("factor", sort=False)):
        colors = [FRAUD if risky else LEGIT for risky in group["is_risky"]]
        ax.bar(group["group"], group["fraud_rate_pct"], color=colors)
        for x, rate in enumerate(group["fraud_rate_pct"]):
            ax.text(x, rate, f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("Fraud rate (%)")
    fig.suptitle("Fraud rate with vs without each risk factor")
    return _save(fig, "07_risk_factors.png")


def plot_correlation_heatmap(df):
    cols = SKEWED_FEATURES + ["hour", "is_online", "is_international",
                              "is_new_merchant", "is_night", TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation between features and is_fraud")
    return _save(fig, "08_correlation_heatmap.png")


def plot_distance_vs_amount(df):
    legit = df[df[TARGET] == 0].sample(min(5000, (df[TARGET] == 0).sum()),
                                       random_state=config.RANDOM_STATE)
    fraud = df[df[TARGET] == 1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(legit["distance_from_home_km"] + 0.1, legit["amount"], s=6, alpha=0.3,
               color=LEGIT, label="Genuine (sample of 5,000)")
    ax.scatter(fraud["distance_from_home_km"] + 0.1, fraud["amount"], s=10, alpha=0.6,
               color=FRAUD, label="Fraud")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from home (km, log scale)")
    ax.set_ylabel("Amount (log scale)")
    ax.set_title("Fraud clusters far from home")
    ax.legend()
    return _save(fig, "09_distance_vs_amount.png")


def plot_hour_weekday_heatmap(df):
    matrix = analysis.hour_weekday_matrix(df)
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(matrix, cmap="Reds", linewidths=0.3, cbar_kws={"label": "Frauds"}, ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    ax.set_title("When does fraud happen? (fraud count by day and hour)")
    return _save(fig, "10_hour_weekday_heatmap.png")


def plot_fraud_types(df):
    types = analysis.fraud_type_summary(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    shares = types["frauds"] / types["frauds"].sum()
    ax1.pie(types["frauds"], labels=[f"{t}\n{s:.0%}" for t, s in zip(types["fraud_type"], shares)],
            startangle=90, colors=sns.color_palette("Reds_r", len(types)),
            wedgeprops={"width": 0.45})
    ax1.set_title("Share of fraud by type")
    ax2.barh(types["fraud_type"], types["avg_amount"], color=FRAUD)
    for y, value in enumerate(types["avg_amount"]):
        ax2.text(value, y, f"  {value:,.0f}", va="center", fontsize=9)
    ax2.set_xlabel("Average amount per fraud")
    ax2.set_title("Average fraud amount by type")
    ax2.set_xlim(0, types["avg_amount"].max() * 1.2)
    return _save(fig, "11_fraud_types.png")


EDA_CHARTS = [
    plot_class_distribution, plot_daily_trend, plot_fraud_by_hour,
    plot_fraud_by_category, plot_amount_distribution, plot_fraud_by_weekday,
    plot_risk_factors, plot_correlation_heatmap, plot_distance_vs_amount,
    plot_hour_weekday_heatmap, plot_fraud_types,
]


def create_eda_charts(df):
    """Draw every EDA chart. `df` must already have add_features() applied."""
    config.ensure_dirs()
    paths = [chart(df) for chart in EDA_CHARTS]
    for path in paths:
        print(f"  Saved chart -> {path.relative_to(config.PROJECT_ROOT)}")
    return paths
