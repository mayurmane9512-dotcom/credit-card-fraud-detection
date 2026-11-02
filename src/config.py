"""Central settings: folder paths, dataset size and model file names.

Every other module imports from here, so changing a value in this one file
(for example N_TRANSACTIONS) changes it for the whole project.
"""
from pathlib import Path

# ---------------------------------------------------------------- folders --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ------------------------------------------------------------------ files --
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"          # full dataset
SAMPLE_CSV = DATA_DIR / "sample_transactions.csv"         # unseen rows for demo
METRICS_JSON = REPORTS_DIR / "metrics.json"               # model scores
TEST_PREDICTIONS_CSV = REPORTS_DIR / "test_predictions.csv"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"

MODEL_FILES = {
    "Random Forest": MODELS_DIR / "random_forest.pkl",
    "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
}

# ------------------------------------------------------- dataset settings --
RANDOM_STATE = 42          # fixed seed -> same data and results every run
N_TRANSACTIONS = 50_000    # rows in the generated dataset
FRAUD_RATE = 0.02          # 2% of transactions are fraud (real life: 0.1-2%)
N_CUSTOMERS = 2_000
START_DATE = "2026-01-01"
N_DAYS = 90                # three months of transactions

# ---------------------------------------------------------- model settings --
TEST_SIZE = 0.2            # 80% train / 20% test
DECISION_THRESHOLD = 0.5   # probability above which a transaction is flagged

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_shopping", "entertainment",
    "health", "travel", "electronics", "clothing", "utilities", "jewelry",
    "atm_withdrawal",
]
CARD_TYPES = ["Visa", "Mastercard", "RuPay", "Amex"]

# Colours used by every chart so "fraud" always looks the same
COLOR_LEGIT = "#2E86AB"
COLOR_FRAUD = "#E63946"
MODEL_COLORS = {"Random Forest": "#2A9D8F", "Logistic Regression": "#F4A261"}


def ensure_dirs():
    """Create the output folders if they do not exist yet."""
    for folder in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
