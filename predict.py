"""Score a CSV file of transactions from the command line.

Usage:
    python predict.py --input data/sample_transactions.csv
    python predict.py --input my_file.csv --model lr --output reports/my_predictions.csv
    python predict.py --input data/sample_transactions.csv --threshold 0.3
"""
import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.predictor import load_model, predict

MODEL_NAMES = {"rf": "Random Forest", "lr": "Logistic Regression"}


def main():
    parser = argparse.ArgumentParser(description="Predict fraud for transactions in a CSV file")
    parser.add_argument("--input", required=True, help="CSV file with transactions")
    parser.add_argument("--model", choices=MODEL_NAMES, default="rf",
                        help="rf = Random Forest (default), lr = Logistic Regression")
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD,
                        help="probability at or above which a transaction is flagged as fraud")
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "predictions.csv"),
                        help="where to save the results (default reports/predictions.csv)")
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model]
    transactions = pd.read_csv(args.input)
    results = predict(transactions, load_model(model_name), args.threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    flagged = results[results["prediction"] == "FRAUD"].sort_values("fraud_probability",
                                                                     ascending=False)
    print(f"\nModel: {model_name} | threshold: {args.threshold}")
    print(f"Scored {len(results)} transactions -> {len(flagged)} flagged as FRAUD\n")
    columns = [c for c in ("transaction_id", "merchant_category", "amount",
                           "fraud_probability", "risk_level") if c in results]
    if len(flagged):
        print(flagged[columns].to_string(index=False))
    print(f"\nFull results saved to {output}")


if __name__ == "__main__":
    main()
