"""Score new transactions with a saved model and explain the result."""
import joblib
import pandas as pd

from src import config
from src.preprocessing import MODEL_FEATURES, add_features, validate_columns

HIGH_RISK_CATEGORIES = {"online_shopping", "electronics", "jewelry", "travel", "atm_withdrawal"}


def models_available():
    return all(path.exists() for path in config.MODEL_FILES.values())


def load_model(name="Random Forest"):
    path = config.MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python main.py' first to train the models.")
    return joblib.load(path)


def risk_level(probability):
    if probability >= 0.7:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


def predict(df, model, threshold=config.DECISION_THRESHOLD):
    """Return `df` with fraud_probability, prediction and risk_level columns added."""
    validate_columns(df)
    features = add_features(df)
    probability = model.predict_proba(features[MODEL_FEATURES])[:, 1]
    out = df.copy()
    out["fraud_probability"] = probability.round(4)
    out["prediction"] = ["FRAUD" if p >= threshold else "Genuine" for p in probability]
    out["risk_level"] = [risk_level(p) for p in probability]
    return out


def red_flags(transaction):
    """Plain-English reasons a transaction looks suspicious (rule-based, for humans).

    `transaction` is one row (dict or Series) after add_features().
    """
    t = transaction
    flags = []
    if t["amount_to_avg_ratio"] >= 3:
        flags.append(f"Amount is {t['amount_to_avg_ratio']:.1f}x the customer's usual spend")
    if t["amount"] < 15 and t["is_new_merchant"]:
        flags.append("Very small charge at a new merchant (possible card-testing)")
    if t["is_night"]:
        flags.append(f"Made late at night ({int(t['hour']):02d}:00)")
    if t["is_international"]:
        flags.append("International transaction")
    if t["distance_from_home_km"] >= 100:
        flags.append(f"{t['distance_from_home_km']:,.0f} km away from the customer's home")
    if t["time_since_last_txn_min"] < 30:
        flags.append(f"Only {t['time_since_last_txn_min']:.0f} min after the previous transaction")
    if t["txn_count_last_24h"] >= 8:
        flags.append(f"{int(t['txn_count_last_24h'])} transactions in the last 24 hours")
    if t["is_new_merchant"]:
        flags.append("First purchase at this merchant")
    if t["failed_attempts"] >= 1:
        flags.append(f"{int(t['failed_attempts'])} failed PIN/CVV attempt(s) before this one")
    if t["merchant_category"] in HIGH_RISK_CATEGORIES:
        flags.append(f"High-risk category: {t['merchant_category']}")
    return flags


def explain(df_row):
    """red_flags() for a raw single-row DataFrame."""
    return red_flags(add_features(pd.DataFrame(df_row)).iloc[0])
