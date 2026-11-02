"""Feature engineering and the scikit-learn preprocessing pipeline.

The same preprocessing is saved *inside* each trained model file, so the
dashboard and predict.py can pass raw transactions straight to the model.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

TARGET = "is_fraud"

# Columns a new transaction must have before we can score it
RAW_INPUT_COLUMNS = [
    "timestamp", "merchant_category", "card_type", "amount", "customer_avg_amount",
    "is_online", "is_international", "distance_from_home_km",
    "time_since_last_txn_min", "txn_count_last_24h", "is_new_merchant",
    "failed_attempts",
]

# Model inputs, grouped by how they are preprocessed
SKEWED_FEATURES = [            # long right tail -> log1p, then scale
    "amount", "customer_avg_amount", "amount_to_avg_ratio",
    "distance_from_home_km", "time_since_last_txn_min",
    "txn_count_last_24h", "failed_attempts",
]
SCALED_FEATURES = ["hour"]     # just scale
BINARY_FEATURES = ["is_online", "is_international", "is_new_merchant",
                   "is_weekend", "is_night"]
CATEGORICAL_FEATURES = ["merchant_category", "card_type"]
MODEL_FEATURES = SKEWED_FEATURES + SCALED_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def validate_columns(df):
    """Raise a clear error if an uploaded file is missing required columns."""
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))


def add_features(df):
    """Derive time and behaviour features from the raw transaction columns."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.day_of_week
    df["day_name"] = pd.Categorical(ts.dt.day_of_week.map(dict(enumerate(DAY_NAMES))),
                                    categories=DAY_NAMES, ordered=True)
    df["date"] = ts.dt.normalize()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    # How unusual is this amount for this customer? (5 = five times their usual spend)
    df["amount_to_avg_ratio"] = (df["amount"] / df["customer_avg_amount"].clip(lower=1)).round(3)
    return df


def build_preprocessor():
    """Impute -> transform -> scale numbers, one-hot encode categories."""
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    scaled = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("skewed", skewed, SKEWED_FEATURES),
        ("scaled", scaled, SCALED_FEATURES),
        ("binary", binary, BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def clean_feature_names(names):
    """'cat__merchant_category_travel' -> 'merchant_category=travel'."""
    cleaned = []
    for name in names:
        name = name.split("__", 1)[-1]
        for col in CATEGORICAL_FEATURES:
            if name.startswith(col + "_"):
                name = f"{col}={name[len(col) + 1:]}"
        cleaned.append(name)
    return cleaned


def split_data(df):
    """Stratified train/test split so both parts keep the same fraud rate."""
    X = df[MODEL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE,
                            stratify=y, random_state=config.RANDOM_STATE)
