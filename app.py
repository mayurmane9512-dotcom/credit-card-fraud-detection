"""Interactive dashboard for the Credit Card Fraud Detection System.

Run with:   streamlit run app.py
Then open:  http://localhost:8501
"""
import json
import threading
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src import analysis, config
from src.predictor import load_model, models_available, predict, red_flags
from src.preprocessing import DAY_NAMES, RAW_INPUT_COLUMNS, add_features

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

FRAUD, LEGIT = config.COLOR_FRAUD, config.COLOR_LEGIT
CLASS_COLORS = {"Fraud": FRAUD, "Genuine": LEGIT}
MODEL_COLORS = config.MODEL_COLORS
MODEL_NAMES = list(config.MODEL_FILES)

COLUMN_DESCRIPTIONS = {
    "transaction_id": "Unique id of the transaction",
    "timestamp": "Date and time of the transaction",
    "customer_id": "Card holder id",
    "card_type": "Card network (Visa, Mastercard, RuPay, Amex)",
    "merchant_category": "Type of shop the card was used at",
    "amount": "Transaction amount",
    "customer_avg_amount": "The customer's usual (historical average) transaction amount",
    "is_online": "1 = online / card-not-present, 0 = in-store",
    "is_international": "1 = merchant is in another country",
    "distance_from_home_km": "Distance between the merchant and the customer's home",
    "time_since_last_txn_min": "Minutes since the customer's previous transaction",
    "txn_count_last_24h": "Number of transactions on the card in the previous 24 hours",
    "is_new_merchant": "1 = first time this card is used at this merchant",
    "failed_attempts": "Wrong PIN / CVV attempts just before this transaction",
    "fraud_type": "For analysis only (not a model input): kind of fraud, 'none' if genuine",
    "is_fraud": "Label: 1 = fraud, 0 = genuine (what the models learn to predict)",
}

FRAUD_TYPE_DESCRIPTIONS = {
    "stolen_card_online": "Stolen card details used online: large amount, new merchant, "
                          "electronics / online shopping, often at night.",
    "card_testing": "A criminal checks whether a stolen card works with many tiny charges "
                    "in quick succession.",
    "cloned_card": "A skimmed / cloned card used in a shop or ATM far from home, often "
                   "abroad, after wrong PIN attempts.",
    "account_takeover": "A criminal controls the account and makes a few very large purchases "
                        "after failed CVV / log-in attempts.",
}


# ------------------------------------------------------------- data access --
@st.cache_data
def load_transactions():
    df = add_features(pd.read_csv(config.TRANSACTIONS_CSV))
    df["class"] = df["is_fraud"].map({0: "Genuine", 1: "Fraud"})
    return df


@st.cache_data
def load_reports():
    with open(config.METRICS_JSON, encoding="utf-8") as fh:
        metrics = json.load(fh)
    return (metrics, pd.read_csv(config.TEST_PREDICTIONS_CSV),
            pd.read_csv(config.FEATURE_IMPORTANCE_CSV))


@st.cache_resource
def get_model(name):
    return load_model(name)


def pipeline_outputs_exist():
    return (config.TRANSACTIONS_CSV.exists() and config.METRICS_JSON.exists()
            and config.TEST_PREDICTIONS_CSV.exists() and models_available())


@st.cache_resource
def _setup_lock():
    """One lock shared by every visitor, so the first-start setup runs only once."""
    return threading.Lock()


def ensure_pipeline_outputs():
    """A fresh install or cloud host has no data/models yet: build them on the first visit."""
    if pipeline_outputs_exist():
        return
    with st.spinner("First start: generating the data and training the models "
                    "(takes up to a minute) ..."), _setup_lock():
        if not pipeline_outputs_exist():  # another visitor may have just built them
            from main import run_pipeline
            run_pipeline()
    st.cache_data.clear()


# ----------------------------------------------------------------- helpers --
def fraud_rate_bar(table, x, title, horizontal=False, overall=None):
    """Bar chart of fraud rate; bars above the overall rate are red."""
    colors = [FRAUD if overall is not None and r > overall else LEGIT
              for r in table["fraud_rate_pct"]]
    if horizontal:
        fig = go.Figure(go.Bar(y=table[x].astype(str), x=table["fraud_rate_pct"], orientation="h",
                               marker_color=colors, customdata=table[["frauds", "transactions"]],
                               hovertemplate="%{y}: %{x:.2f}%<br>%{customdata[0]} of "
                                             "%{customdata[1]:,} transactions<extra></extra>"))
        fig.update_layout(xaxis_title="Fraud rate (%)")
    else:
        fig = go.Figure(go.Bar(x=table[x].astype(str), y=table["fraud_rate_pct"],
                               marker_color=colors, customdata=table[["frauds", "transactions"]],
                               hovertemplate="%{x}: %{y:.2f}%<br>%{customdata[0]} of "
                                             "%{customdata[1]:,} transactions<extra></extra>"))
        fig.update_layout(yaxis_title="Fraud rate (%)")
    if overall is not None:
        line = dict(line_dash="dash", line_color="grey", annotation_text="overall")
        # a plain if-statement: a bare "a if c else b" line is auto-displayed by Streamlit magic
        if horizontal:
            fig.add_vline(x=overall, annotation_position="bottom right", **line)
        else:
            fig.add_hline(y=overall, **line)
    fig.update_layout(title=title, height=420, margin=dict(t=60, b=40))
    return fig


def gauge(probability, threshold, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#333333"},
            "steps": [{"range": [0, 30], "color": "#C8E6C9"},
                      {"range": [30, 70], "color": "#FFE0B2"},
                      {"range": [70, 100], "color": "#FFCDD2"}],
            "threshold": {"line": {"color": FRAUD, "width": 4}, "value": threshold * 100},
        }))
    fig.update_layout(height=260, margin=dict(t=60, b=10, l=30, r=30))
    return fig


# ------------------------------------------------------------------- pages --
def page_overview(df, metrics):
    st.title("💳 Credit Card Fraud Detection System")
    st.caption("Identify suspicious transactions with machine learning and explore fraud patterns.")

    s = analysis.summary_stats(df)
    best = metrics["best_model"]
    c = st.columns(5)
    c[0].metric("Transactions", f"{s['transactions']:,}")
    c[1].metric("Fraudulent", f"{s['frauds']:,}")
    c[2].metric("Fraud rate", f"{s['fraud_rate_pct']:.2f}%")
    c[3].metric("Amount lost to fraud", f"{s['fraud_amount']:,.0f}")
    c[4].metric("Best model PR-AUC", f"{metrics['models'][best]['pr_auc']:.3f}", help=best)

    left, right = st.columns([1, 2])
    counts = df["class"].value_counts().rename_axis("class").reset_index(name="count")
    pie = px.pie(counts, names="class", values="count", color="class", hole=0.55,
                 color_discrete_map=CLASS_COLORS, title="Genuine vs fraud transactions")
    pie.update_traces(textinfo="percent+label")
    left.plotly_chart(pie)

    daily = analysis.daily_trend(df)
    trend = go.Figure()
    trend.add_bar(x=daily["date"], y=daily["frauds"], name="Frauds per day",
                  marker_color=FRAUD, opacity=0.35)
    trend.add_scatter(x=daily["date"], y=daily["frauds_7d_avg"], name="7-day average",
                      line=dict(color=FRAUD, width=3))
    trend.update_layout(title="Daily fraud trend", yaxis_title="Fraud transactions",
                        legend=dict(orientation="h", y=1.08))
    right.plotly_chart(trend)

    st.subheader("How the system works")
    steps = st.columns(4)
    steps[0].info("**1. Data**\n\n50,000 simulated card transactions with realistic fraud patterns.")
    steps[1].info("**2. Explore**\n\nCharts reveal when, where and how fraud happens.")
    steps[2].info("**3. Learn**\n\nRandom Forest and Logistic Regression learn to spot fraud.")
    steps[3].info("**4. Detect**\n\nScore single transactions or whole CSV files instantly.")

    with st.expander("Preview the dataset"):
        st.dataframe(df[list(COLUMN_DESCRIPTIONS)].head(200), hide_index=True)
    with st.expander("What each column means"):
        st.table(pd.DataFrame(COLUMN_DESCRIPTIONS.items(), columns=["Column", "Meaning"]))


def page_trends(df):
    st.title("📈 Fraud Trends & Transaction Patterns")

    with st.sidebar:
        st.subheader("Filters")
        first, last = df["date"].min().date(), df["date"].max().date()
        picked = st.date_input("Date range", (first, last), min_value=first, max_value=last)
        categories = st.multiselect("Merchant categories", config.MERCHANT_CATEGORIES,
                                    default=config.MERCHANT_CATEGORIES)
        channel = st.radio("Channel", ["All", "Online", "In-store"], horizontal=True)

    start, end = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2
                  else (first, last))
    view = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)
              & df["merchant_category"].isin(categories)]
    if channel != "All":
        view = view[view["is_online"] == (1 if channel == "Online" else 0)]
    if view.empty or view["is_fraud"].sum() == 0:
        st.info("No fraud in the selected filters - widen the date range or pick more categories.")
        return

    s = analysis.summary_stats(view)
    overall = s["fraud_rate_pct"]
    c = st.columns(4)
    c[0].metric("Transactions", f"{s['transactions']:,}")
    c[1].metric("Frauds", f"{s['frauds']:,}", f"{overall:.2f}% fraud rate", delta_color="off")
    c[2].metric("Avg fraud amount", f"{s['avg_fraud_amount']:,.0f}")
    c[3].metric("Avg genuine amount", f"{s['avg_legit_amount']:,.0f}")

    tabs = st.tabs(["Over time", "Time of day", "Merchant & card", "Amounts", "Risk factors",
                    "Fraud types"])

    with tabs[0]:
        daily = analysis.daily_trend(view)
        fig = go.Figure()
        fig.add_bar(x=daily["date"], y=daily["frauds"], name="Frauds per day",
                    marker_color=FRAUD, opacity=0.35)
        fig.add_scatter(x=daily["date"], y=daily["frauds_7d_avg"], name="7-day average",
                        line=dict(color=FRAUD, width=3))
        fig.update_layout(title="Daily fraud count", yaxis_title="Frauds",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig)
        left, right = st.columns(2)
        rate = px.line(daily, x="date", y="fraud_rate_pct", title="Daily fraud rate (%)",
                       labels={"fraud_rate_pct": "Fraud rate (%)", "date": "Date"})
        rate.update_traces(line_color="#6C4AB6")
        left.plotly_chart(rate)
        weekly = analysis.weekly_fraud_amount(view)
        money = px.bar(weekly, x="week_start", y="fraud_amount", title="Amount lost to fraud per week",
                       labels={"week_start": "Week starting", "fraud_amount": "Fraud amount"},
                       color_discrete_sequence=[FRAUD])
        right.plotly_chart(money)
        st.caption("The rising trend and the spike in late February simulate a growing fraud "
                   "problem and an organised 'fraud ring' attack.")

    with tabs[1]:
        by_hour = analysis.fraud_rate_by(view, "hour").set_index("hour").reindex(range(24), fill_value=0)
        fig = go.Figure()
        fig.add_bar(x=by_hour.index, y=by_hour["transactions"], name="All transactions",
                    marker_color=LEGIT, opacity=0.35, yaxis="y")
        fig.add_scatter(x=by_hour.index, y=by_hour["fraud_rate_pct"], name="Fraud rate (%)",
                        line=dict(color=FRAUD, width=3), yaxis="y2")
        fig.update_layout(title="Transactions and fraud rate by hour of day",
                          xaxis=dict(title="Hour", dtick=1),
                          yaxis=dict(title="Transactions"),
                          yaxis2=dict(title="Fraud rate (%)", overlaying="y", side="right",
                                      showgrid=False, tickmode="auto", rangemode="tozero"),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
        left, right = st.columns([3, 2])
        matrix = analysis.hour_weekday_matrix(view)
        heat = px.imshow(matrix, aspect="auto", color_continuous_scale="Reds",
                         labels=dict(x="Hour of day", y="", color="Frauds"),
                         title="Fraud count by day and hour")
        left.plotly_chart(heat)
        by_day = analysis.fraud_rate_by(view, "day_name").set_index("day_name").reindex(DAY_NAMES)
        right.plotly_chart(fraud_rate_bar(by_day.reset_index(), "day_name",
                                          "Fraud rate by day of week", overall=overall))

    with tabs[2]:
        left, right = st.columns(2)
        by_cat = analysis.fraud_rate_by(view, "merchant_category").sort_values("fraud_rate_pct")
        left.plotly_chart(fraud_rate_bar(by_cat, "merchant_category",
                                         "Fraud rate by merchant category",
                                         horizontal=True, overall=overall))
        fraud_view = view[view["is_fraud"] == 1]
        tree = px.treemap(fraud_view, path=["merchant_category"], values="amount",
                          title="Where the fraud money goes (amount by category)",
                          color_discrete_sequence=px.colors.sequential.Reds_r)
        tree.update_traces(textinfo="label+percent root")
        right.plotly_chart(tree)
        by_card = analysis.fraud_rate_by(view, "card_type")
        st.plotly_chart(fraud_rate_bar(by_card, "card_type",
                                       "Fraud rate by card type (similar for all brands)",
                                       overall=overall))

    with tabs[3]:
        left, right = st.columns(2)
        box = px.box(view, x="class", y="amount", color="class", log_y=True, points=False,
                     color_discrete_map=CLASS_COLORS, title="Transaction amount (log scale)",
                     category_orders={"class": ["Genuine", "Fraud"]})
        left.plotly_chart(box)
        ratio = px.box(view, x="class", y="amount_to_avg_ratio", color="class", log_y=True,
                       points=False, color_discrete_map=CLASS_COLORS,
                       title="Amount vs the customer's usual spend (log scale)",
                       labels={"amount_to_avg_ratio": "amount / usual amount"},
                       category_orders={"class": ["Genuine", "Fraud"]})
        right.plotly_chart(ratio)
        sample = pd.concat([view[view["is_fraud"] == 0].sample(min(4000, (view["is_fraud"] == 0).sum()),
                                                               random_state=1),
                            view[view["is_fraud"] == 1]])
        scatter = px.scatter(sample, x="distance_from_home_km", y="amount", color="class",
                             log_x=True, log_y=True, opacity=0.5, color_discrete_map=CLASS_COLORS,
                             hover_data=["merchant_category", "hour", "is_international"],
                             title="Distance from home vs amount (all frauds + 4,000 genuine)",
                             labels={"distance_from_home_km": "Distance from home (km)"})
        st.plotly_chart(scatter)

    with tabs[4]:
        rates = analysis.risk_flag_rates(view)
        rates["Risk factor present"] = rates["is_risky"].map({True: "Yes", False: "No"})
        rates["factor"] = rates["factor"].map({"is_online": "Channel", "is_international": "Location",
                                               "is_new_merchant": "Merchant", "is_night": "Time of day"})
        fig = px.bar(rates, x="group", y="fraud_rate_pct", color="Risk factor present",
                     facet_col="factor", text_auto=".1f",
                     color_discrete_map={"Yes": FRAUD, "No": LEGIT},
                     labels={"group": "", "fraud_rate_pct": "Fraud rate (%)"},
                     title="Fraud rate with vs without each risk factor")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_xaxes(matches=None)
        st.plotly_chart(fig)
        left, right = st.columns(2)
        failed = analysis.fraud_rate_by(view.assign(failed=view["failed_attempts"].clip(upper=4)
                                                    .astype(str).replace("4", "4+")), "failed")
        left.plotly_chart(fraud_rate_bar(failed, "failed", "Fraud rate by failed PIN/CVV attempts",
                                         overall=overall))
        bins = pd.cut(view["txn_count_last_24h"], [-1, 2, 5, 8, 12, np.inf],
                      labels=["0-2", "3-5", "6-8", "9-12", "13+"])
        velocity = analysis.fraud_rate_by(view.assign(velocity=bins), "velocity")
        right.plotly_chart(fraud_rate_bar(velocity, "velocity",
                                          "Fraud rate by transactions in last 24 h",
                                          overall=overall))

    with tabs[5]:
        types = analysis.fraud_type_summary(view)
        left, right = st.columns(2)
        donut = px.pie(types, names="fraud_type", values="frauds", hole=0.5,
                       title="Share of fraud by type",
                       color_discrete_sequence=px.colors.sequential.Reds_r)
        left.plotly_chart(donut)
        avg = px.bar(types.sort_values("avg_amount"), x="avg_amount", y="fraud_type",
                     orientation="h", text_auto=",.0f", color_discrete_sequence=[FRAUD],
                     title="Average amount per fraud, by type",
                     labels={"avg_amount": "Average amount", "fraud_type": ""})
        right.plotly_chart(avg)
        weekly_types = (view[view["is_fraud"] == 1]
                        .assign(week=lambda d: d["date"].dt.to_period("W").dt.start_time)
                        .groupby(["week", "fraud_type"]).size().reset_index(name="frauds"))
        st.plotly_chart(px.area(weekly_types, x="week", y="frauds", color="fraud_type",
                                title="Fraud types over time (per week)",
                                color_discrete_sequence=px.colors.sequential.Reds_r,
                                labels={"week": "Week", "frauds": "Frauds"}))
        st.table(pd.DataFrame(FRAUD_TYPE_DESCRIPTIONS.items(), columns=["Fraud type", "Pattern"]))


def page_models(metrics, preds, importance):
    st.title("🤖 Model Performance")
    st.caption(f"Scores on the test set: {metrics['test_rows']:,} transactions the models never "
               f"saw during training. Decision threshold = {metrics['threshold']}.")

    table = pd.DataFrame([
        {"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
         "Recall": m["recall"], "F1-score": m["f1"], "ROC-AUC": m["roc_auc"],
         "PR-AUC": m["pr_auc"], "Train time (s)": m["train_seconds"]}
        for name, m in metrics["models"].items()])
    st.dataframe(table, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in table.columns if c != "Model"})
    st.success(f"Best model (highest PR-AUC): **{metrics['best_model']}**")

    with st.expander("What do these metrics mean?"):
        st.markdown("""
- **Accuracy** - share of all predictions that are correct. *Misleading here:* a model that
  says "genuine" for everything is already 98% accurate because only 2% is fraud.
- **Precision** - of the transactions we flagged, how many were really fraud? (low = many false alarms)
- **Recall** - of all real frauds, how many did we catch? (low = fraud slips through)
- **F1-score** - balance of precision and recall.
- **ROC-AUC** - how well the model ranks fraud above genuine (1.0 = perfect, 0.5 = random).
- **PR-AUC** - like ROC-AUC but focused on the rare fraud class; the best single score for
  imbalanced data.
""")

    long = table.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall",
                                                    "F1-score", "ROC-AUC", "PR-AUC"],
                      var_name="Metric", value_name="Score")
    bar = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", text_auto=".2f",
                 color_discrete_map=MODEL_COLORS, title="Model comparison", range_y=[0, 1.08])
    st.plotly_chart(bar)

    st.subheader("Try a different decision threshold")
    left, right = st.columns([1, 2])
    with left:
        model_name = st.radio("Model", MODEL_NAMES, key="perf_model")
        threshold = st.slider("Flag as fraud when probability is at least", 0.05, 0.95,
                              float(metrics["threshold"]), 0.05)
        st.caption("Lower threshold -> catch more fraud (recall up) but more false alarms "
                   "(precision down). Banks choose based on the cost of each mistake.")
    prob = preds["prob_" + model_name.lower().replace(" ", "_")]
    actual = preds["y_true"] == 1
    flagged = prob >= threshold
    tp, fp = int((flagged & actual).sum()), int((flagged & ~actual).sum())
    fn, tn = int((~flagged & actual).sum()), int((~flagged & ~actual).sum())
    caught = preds.loc[flagged & actual, "amount"].sum()
    total_fraud = preds.loc[actual, "amount"].sum()
    with right:
        m = st.columns(4)
        m[0].metric("Precision", f"{tp / max(tp + fp, 1):.1%}")
        m[1].metric("Recall", f"{tp / max(tp + fn, 1):.1%}")
        m[2].metric("False alarms", f"{fp:,}")
        m[3].metric("Fraud amount caught", f"{caught / max(total_fraud, 1):.1%}")
        cm = px.imshow([[tn, fp], [fn, tp]], text_auto=",d", color_continuous_scale="Blues",
                       x=["Predicted genuine", "Predicted fraud"],
                       y=["Actual genuine", "Actual fraud"], title=f"Confusion matrix - {model_name}")
        cm.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(cm)

    left, right = st.columns(2)
    roc, pr = go.Figure(), go.Figure()
    for name in MODEL_NAMES:
        p = preds["prob_" + name.lower().replace(" ", "_")]
        fpr, tpr, _ = roc_curve(preds["y_true"], p)
        roc.add_scatter(x=fpr, y=tpr, name=f"{name} (AUC {roc_auc_score(preds['y_true'], p):.3f})",
                        line=dict(color=MODEL_COLORS[name], width=3))
        precision, recall, _ = precision_recall_curve(preds["y_true"], p)
        pr.add_scatter(x=recall, y=precision,
                       name=f"{name} (AP {average_precision_score(preds['y_true'], p):.3f})",
                       line=dict(color=MODEL_COLORS[name], width=3))
    roc.add_scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(dash="dash", color="grey"))
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate (recall)", legend=dict(x=0.35, y=0.08))
    pr.update_layout(title="Precision-Recall curve", xaxis_title="Recall",
                     yaxis_title="Precision", legend=dict(x=0.02, y=0.08))
    left.plotly_chart(roc)
    right.plotly_chart(pr)

    st.subheader("Which features matter most?")
    left, right = st.columns(2)
    top_rf = importance.head(15).iloc[::-1]
    left.plotly_chart(px.bar(top_rf, x="rf_importance", y="feature", orientation="h",
                             color_discrete_sequence=[MODEL_COLORS["Random Forest"]],
                             title="Random Forest - feature importance",
                             labels={"rf_importance": "Importance", "feature": ""}))
    top_lr = importance.reindex(importance["lr_coefficient"].abs()
                                .sort_values(ascending=False).index).head(15).iloc[::-1]
    top_lr = top_lr.assign(effect=np.where(top_lr["lr_coefficient"] > 0,
                                           "Raises fraud risk", "Lowers fraud risk"))
    right.plotly_chart(px.bar(top_lr, x="lr_coefficient", y="feature", color="effect",
                              orientation="h", title="Logistic Regression - coefficients",
                              color_discrete_map={"Raises fraud risk": FRAUD,
                                                  "Lowers fraud risk": LEGIT},
                              labels={"lr_coefficient": "Coefficient", "feature": ""}))


PRESETS = {
    "genuine": {
        "amount": 45.0, "customer_avg_amount": 60.0, "merchant_category": "grocery",
        "card_type": "Visa", "txn_date": date(2026, 3, 10), "txn_hour": 14,
        "distance_from_home_km": 3.0, "time_since_last_txn_min": 480.0,
        "txn_count_last_24h": 2, "failed_attempts": 0, "is_online": False,
        "is_international": False, "is_new_merchant": False,
    },
    "suspicious": {
        "amount": 890.0, "customer_avg_amount": 55.0, "merchant_category": "electronics",
        "card_type": "Mastercard", "txn_date": date(2026, 3, 14), "txn_hour": 2,
        "distance_from_home_km": 1450.0, "time_since_last_txn_min": 6.0,
        "txn_count_last_24h": 11, "failed_attempts": 2, "is_online": True,
        "is_international": True, "is_new_merchant": True,
    },
}


def _load_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state["f_" + key] = value


def page_check(threshold):
    st.title("🔍 Check a Single Transaction")
    st.caption("Enter the transaction details. Both models estimate the probability that it is fraud.")
    for key, value in PRESETS["genuine"].items():
        st.session_state.setdefault("f_" + key, value)

    b = st.columns([1, 1, 3])
    b[0].button("Load genuine example", on_click=_load_preset, args=("genuine",))
    b[1].button("Load suspicious example", on_click=_load_preset, args=("suspicious",))

    with st.form("transaction"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Purchase**")
            st.number_input("Amount", min_value=1.0, step=10.0, key="f_amount")
            st.selectbox("Merchant category", config.MERCHANT_CATEGORIES, key="f_merchant_category")
            st.selectbox("Card type", config.CARD_TYPES, key="f_card_type")
            st.checkbox("Online purchase", key="f_is_online")
        with c2:
            st.markdown("**When & where**")
            st.date_input("Date", key="f_txn_date")
            st.slider("Hour of day", 0, 23, key="f_txn_hour")
            st.number_input("Distance from home (km)", min_value=0.0, step=10.0,
                            key="f_distance_from_home_km")
            st.checkbox("International merchant", key="f_is_international")
        with c3:
            st.markdown("**Customer behaviour**")
            st.number_input("Customer's usual amount", min_value=1.0, step=5.0,
                            key="f_customer_avg_amount")
            st.number_input("Minutes since previous transaction", min_value=0.0, step=5.0,
                            key="f_time_since_last_txn_min")
            st.number_input("Transactions in last 24 hours", min_value=0, step=1,
                            key="f_txn_count_last_24h")
            st.number_input("Failed PIN/CVV attempts", min_value=0, step=1, key="f_failed_attempts")
            st.checkbox("First purchase at this merchant", key="f_is_new_merchant")
        st.form_submit_button("Check for fraud", type="primary")

    s = st.session_state
    when = datetime.combine(s.f_txn_date, time(s.f_txn_hour, 0))
    transaction = pd.DataFrame([{
        "timestamp": when.strftime("%Y-%m-%d %H:%M:%S"),
        "merchant_category": s.f_merchant_category, "card_type": s.f_card_type,
        "amount": s.f_amount, "customer_avg_amount": s.f_customer_avg_amount,
        "is_online": int(s.f_is_online), "is_international": int(s.f_is_international),
        "distance_from_home_km": s.f_distance_from_home_km,
        "time_since_last_txn_min": s.f_time_since_last_txn_min,
        "txn_count_last_24h": s.f_txn_count_last_24h, "is_new_merchant": int(s.f_is_new_merchant),
        "failed_attempts": s.f_failed_attempts,
    }])

    st.subheader("Result")
    cols = st.columns(2)
    for col, name in zip(cols, MODEL_NAMES):
        result = predict(transaction, get_model(name), threshold).iloc[0]
        with col:
            st.plotly_chart(gauge(result["fraud_probability"], threshold, name), key=f"gauge_{name}")
            message = f"**{result['prediction']}** - {result['risk_level']} risk " \
                      f"({result['fraud_probability']:.1%} fraud probability)"
            (st.error if result["prediction"] == "FRAUD" else st.success)(message)

    flags = red_flags(add_features(transaction).iloc[0])
    st.markdown("**Red flags in this transaction**")
    if flags:
        st.markdown("\n".join(f"- ⚠️ {flag}" for flag in flags))
    else:
        st.markdown("- ✅ No obvious red flags")
    st.caption("Red flags are simple rules to help a human analyst understand the case. "
               "The models look at all features together, so they can disagree with the rules.")


def page_batch(threshold):
    st.title("📂 Batch Prediction")
    st.caption("Upload a CSV file of transactions and score all of them at once.")

    with st.expander("Required CSV columns / download a sample file"):
        st.code(", ".join(RAW_INPUT_COLUMNS))
        st.download_button("Download sample CSV", config.SAMPLE_CSV.read_bytes(),
                           "sample_transactions.csv", "text/csv")

    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    model_name = st.radio("Model", MODEL_NAMES, horizontal=True, key="batch_model")
    if uploaded is None:
        st.info("No file uploaded - showing results for the bundled demo file "
                "data/sample_transactions.csv (30 new transactions the models have never seen).")

    try:
        transactions = pd.read_csv(uploaded if uploaded is not None else config.SAMPLE_CSV)
        results = predict(transactions, get_model(model_name), threshold)
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"Could not score this file: {exc}")
        return

    results = results.sort_values("fraud_probability", ascending=False, ignore_index=True)
    flagged = results[results["prediction"] == "FRAUD"]
    c = st.columns(4)
    c[0].metric("Transactions scored", f"{len(results):,}")
    c[1].metric("Flagged as fraud", f"{len(flagged):,}")
    c[2].metric("High risk", f"{(results['risk_level'] == 'High').sum():,}")
    c[3].metric("Flagged amount", f"{flagged['amount'].sum():,.0f}")

    first = ["transaction_id", "prediction", "fraud_probability", "risk_level"]
    ordered = [col for col in first if col in results] + \
              [col for col in results.columns if col not in first]
    st.dataframe(results[ordered], hide_index=True, column_config={
        "fraud_probability": st.column_config.ProgressColumn(
            "Fraud probability", min_value=0.0, max_value=1.0, format="%.2f"),
    })
    st.download_button("Download results as CSV", results[ordered].to_csv(index=False).encode("utf-8"),
                       "fraud_predictions.csv", "text/csv", type="primary")

    hist = px.histogram(results, x="fraud_probability", nbins=20, color="prediction",
                        color_discrete_map={"FRAUD": FRAUD, "Genuine": LEGIT},
                        title="Distribution of fraud probabilities",
                        labels={"fraud_probability": "Fraud probability"})
    hist.add_vline(x=threshold, line_dash="dash", annotation_text="threshold")
    st.plotly_chart(hist)


# -------------------------------------------------------------------- main --
def main():
    ensure_pipeline_outputs()

    df = load_transactions()
    metrics, preds, importance = load_reports()

    st.sidebar.title("💳 Fraud Detection")
    page = st.sidebar.radio("Go to", ["🏠 Overview", "📈 Fraud Trends", "🤖 Model Performance",
                                      "🔍 Check a Transaction", "📂 Batch Prediction"])
    threshold = config.DECISION_THRESHOLD
    if page in ("🔍 Check a Transaction", "📂 Batch Prediction"):
        threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95,
                                      float(config.DECISION_THRESHOLD), 0.05,
                                      help="Transactions with a fraud probability at or above "
                                           "this value are flagged as FRAUD.")
    st.sidebar.divider()
    st.sidebar.caption("Models: Random Forest & Logistic Regression  \n"
                       "Data: synthetic card transactions  \n"
                       "Details: PROJECT_EXPLANATION.md")

    if page == "🏠 Overview":
        page_overview(df, metrics)
    elif page == "📈 Fraud Trends":
        page_trends(df)
    elif page == "🤖 Model Performance":
        page_models(metrics, preds, importance)
    elif page == "🔍 Check a Transaction":
        page_check(threshold)
    else:
        page_batch(threshold)


main()
