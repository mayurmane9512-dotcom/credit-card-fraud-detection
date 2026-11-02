"""Generate a realistic synthetic credit card transaction dataset.

Real bank transaction data is confidential, so this module simulates it.

Genuine transactions follow each customer's everyday habits, plus two kinds
of unusual-but-honest behaviour that cause false alarms in real systems:
customers on holiday abroad and occasional big planned purchases.

Fraud is generated from four common fraud types (modus operandi). Each type
is unusual in only *some* features and looks normal in the rest, which is
why a model has to combine several signals to catch it:

* stolen_card_online - stolen card details used online: big amount, new
  merchant, electronics / online shopping, often at night.
* card_testing       - criminals check a stolen card with many tiny charges
  in quick succession.
* cloned_card        - a skimmed/cloned card used in a shop or ATM far from
  home, often abroad, after wrong PIN attempts.
* account_takeover   - a criminal controls the account and makes a few very
  large purchases after failed log-in / CVV attempts.
"""
import numpy as np
import pandas as pd

from src import config

# Share of each merchant category for genuine transactions
CATEGORY_P_LEGIT = {
    "grocery": 0.22, "restaurant": 0.15, "gas_station": 0.12,
    "online_shopping": 0.14, "entertainment": 0.07, "health": 0.06,
    "travel": 0.04, "electronics": 0.05, "clothing": 0.08, "utilities": 0.05,
    "jewelry": 0.01, "atm_withdrawal": 0.01,
}

# fraud type -> (share of all fraud, merchant categories it targets)
FRAUD_TYPES = {
    "stolen_card_online": (0.35, {"online_shopping": 0.5, "electronics": 0.35, "travel": 0.15}),
    "card_testing": (0.25, {"online_shopping": 0.5, "entertainment": 0.2,
                            "gas_station": 0.2, "restaurant": 0.1}),
    "cloned_card": (0.25, {"atm_withdrawal": 0.35, "jewelry": 0.25, "electronics": 0.2,
                           "clothing": 0.1, "gas_station": 0.1}),
    "account_takeover": (0.15, {"travel": 0.3, "electronics": 0.3, "jewelry": 0.2,
                                "online_shopping": 0.2}),
}

# Relative number of transactions per hour of day (index 0 = midnight)
HOUR_W_LEGIT = [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.5, 3, 4.5, 5, 5.5, 6,
                6.5, 6, 5.5, 5.5, 6, 6.5, 7, 6.5, 5.5, 4, 3, 2]
HOUR_W_FRAUD = [6, 7, 7.5, 7, 6, 4, 2.5, 2, 2, 2.5, 3, 3,
                3.5, 3, 3, 3, 3, 3.5, 4, 4, 4.5, 5, 5.5, 6]

COLUMN_ORDER = [
    "transaction_id", "timestamp", "customer_id", "card_type", "merchant_category",
    "amount", "customer_avg_amount", "is_online", "is_international",
    "distance_from_home_km", "time_since_last_txn_min", "txn_count_last_24h",
    "is_new_merchant", "failed_attempts", "fraud_type", "is_fraud",
]


def _normalise(weights):
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def _choose(rng, options, size):
    """Pick `size` keys from a {value: probability} dict."""
    return rng.choice(list(options), size=size, p=_normalise(list(options.values())))


def _everyday_behaviour(n, avg_spend, rng):
    """Feature values of ordinary, honest transactions."""
    return {
        "amount": avg_spend * rng.lognormal(0, 0.5, n),
        "hour": rng.choice(24, size=n, p=_normalise(HOUR_W_LEGIT)),
        "is_online": (rng.random(n) < 0.15).astype(int),
        "is_international": (rng.random(n) < 0.02).astype(int),
        "distance_from_home_km": rng.exponential(10, n),
        "time_since_last_txn_min": rng.exponential(600, n),
        "txn_count_last_24h": rng.poisson(3, n),
        "is_new_merchant": (rng.random(n) < 0.12).astype(int),
        "failed_attempts": rng.poisson(0.05, n),
    }


def _override(features, rows, **values):
    """Replace feature values for the selected rows only."""
    for key, value in values.items():
        features[key][rows] = value


def _genuine(n, avg_spend, rng):
    category = _choose(rng, CATEGORY_P_LEGIT, n)
    f = _everyday_behaviour(n, avg_spend, rng)

    # Customers on holiday: far from home and often abroad, but honest
    away = (category == "travel") | (rng.random(n) < 0.03)
    k = away.sum()
    _override(f, away, distance_from_home_km=rng.exponential(800, k) + 50,
              is_international=rng.random(k) < 0.5)

    # Occasional big planned purchases (a new phone, flight tickets ...)
    big = rng.random(n) < 0.03
    _override(f, big, amount=avg_spend[big] * rng.lognormal(1.3, 0.4, big.sum()))
    return category, f, np.full(n, "none", dtype=object)


def _fraud(n, avg_spend, rng):
    names = list(FRAUD_TYPES)
    fraud_type = rng.choice(names, size=n, p=_normalise([FRAUD_TYPES[t][0] for t in names]))
    category = np.empty(n, dtype=object)
    for name in names:
        rows = fraud_type == name
        category[rows] = _choose(rng, FRAUD_TYPES[name][1], rows.sum())

    f = _everyday_behaviour(n, avg_spend, rng)   # start normal, then add the fraud signals

    def night(k):
        return rng.choice(24, size=k, p=_normalise(HOUR_W_FRAUD))

    rows = fraud_type == "stolen_card_online"
    k = rows.sum()
    _override(f, rows, amount=avg_spend[rows] * rng.lognormal(1.0, 0.5, k), hour=night(k),
              is_online=1, is_international=rng.random(k) < 0.25,
              is_new_merchant=rng.random(k) < 0.85)

    rows = fraud_type == "card_testing"
    k = rows.sum()
    _override(f, rows, amount=rng.uniform(1, 10, k), hour=night(k),
              is_online=rng.random(k) < 0.8, time_since_last_txn_min=rng.exponential(8, k),
              txn_count_last_24h=rng.poisson(10, k), is_new_merchant=rng.random(k) < 0.9)

    rows = fraud_type == "cloned_card"
    k = rows.sum()
    _override(f, rows, amount=avg_spend[rows] * rng.lognormal(0.9, 0.6, k), hour=night(k),
              is_online=0, is_international=rng.random(k) < 0.7,
              distance_from_home_km=rng.exponential(1500, k) + 100,
              is_new_merchant=rng.random(k) < 0.7, failed_attempts=rng.poisson(1.5, k))

    rows = fraud_type == "account_takeover"
    k = rows.sum()
    _override(f, rows, amount=avg_spend[rows] * rng.lognormal(1.6, 0.5, k),
              is_online=rng.random(k) < 0.6, time_since_last_txn_min=rng.exponential(120, k),
              is_new_merchant=rng.random(k) < 0.8, failed_attempts=rng.poisson(2.0, k))
    return category, f, fraud_type


def _day_weights(is_fraud, n_days, start):
    """How likely each day is: fraud grows over time and has a 'fraud ring' week."""
    days = np.arange(n_days)
    weekday = (pd.Timestamp(start).day_of_week + days) % 7
    weekend = weekday >= 5
    if is_fraud:
        weights = 1 + 0.8 * days / n_days + 0.3 * weekend
        weights[(days >= 52) & (days <= 58)] += 1.5
    else:
        weights = 1 + 0.15 * weekend
    return _normalise(weights)


def _simulate_block(n, is_fraud, rng, cust_avg, cust_card, start, n_days):
    customer = rng.integers(0, len(cust_avg), size=n)
    avg_spend = cust_avg[customer]
    category, f, fraud_type = (_fraud if is_fraud else _genuine)(n, avg_spend, rng)

    # Physical rules: online shops are always online, ATMs never are
    f["is_online"] = np.where(category == "online_shopping", 1, f["is_online"])
    f["is_online"] = np.where(category == "atm_withdrawal", 0, f["is_online"])

    day = rng.choice(n_days, size=n, p=_day_weights(is_fraud, n_days, start))
    timestamp = (
        pd.Timestamp(start)
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(f.pop("hour"), unit="h")
        + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
    )

    block = pd.DataFrame({
        "timestamp": timestamp,
        "customer_id": [f"C{c:04d}" for c in customer],
        "card_type": cust_card[customer],
        "merchant_category": category,
        "customer_avg_amount": avg_spend,
        **f,
    })
    block["amount"] = block["amount"].clip(lower=1.0)
    block["fraud_type"] = fraud_type
    block["is_fraud"] = int(is_fraud)
    return block


def generate_transactions(n_transactions=config.N_TRANSACTIONS,
                          fraud_rate=config.FRAUD_RATE,
                          n_customers=config.N_CUSTOMERS,
                          seed=config.RANDOM_STATE,
                          start=config.START_DATE,
                          n_days=config.N_DAYS):
    """Return a DataFrame of simulated card transactions with an is_fraud label."""
    rng = np.random.default_rng(seed)

    # Each customer has a usual spending level and one card brand
    cust_avg = rng.lognormal(mean=np.log(60), sigma=0.6, size=n_customers)
    cust_card = rng.choice(config.CARD_TYPES, size=n_customers, p=[0.40, 0.30, 0.20, 0.10])

    n_fraud = int(round(n_transactions * fraud_rate))
    blocks = [
        _simulate_block(n_transactions - n_fraud, False, rng, cust_avg, cust_card, start, n_days),
        _simulate_block(n_fraud, True, rng, cust_avg, cust_card, start, n_days),
    ]
    df = pd.concat(blocks, ignore_index=True).sort_values("timestamp", ignore_index=True)

    df.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(1, len(df) + 1)])
    df = df[COLUMN_ORDER].copy()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ("amount", "customer_avg_amount", "distance_from_home_km"):
        df[col] = df[col].round(2)
    df["time_since_last_txn_min"] = df["time_since_last_txn_min"].round(1)
    return df


def save_datasets(n_transactions=config.N_TRANSACTIONS, fraud_rate=config.FRAUD_RATE):
    """Create data/transactions.csv (training data) and data/sample_transactions.csv.

    The sample file is generated with a different seed, so its rows were never
    seen during training. Like real incoming transactions it has no label
    (is_fraud / fraud_type); it is used to demo batch prediction.
    """
    config.ensure_dirs()
    df = generate_transactions(n_transactions, fraud_rate)
    df.to_csv(config.TRANSACTIONS_CSV, index=False)

    sample = generate_transactions(30, fraud_rate=0.3, seed=config.RANDOM_STATE + 1)
    sample.drop(columns=["is_fraud", "fraud_type"]).sample(frac=1, random_state=1).to_csv(
        config.SAMPLE_CSV, index=False)

    print(f"  Saved {len(df):,} transactions ({df['is_fraud'].sum():,} fraud, "
          f"{df['is_fraud'].mean():.2%}) -> {config.TRANSACTIONS_CSV}")
    print(f"  Saved {len(sample)} unlabelled demo transactions -> {config.SAMPLE_CSV}")
    return df
