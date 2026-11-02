"""Run the full fraud detection pipeline: dataset -> charts -> trained models.

Usage:
    python main.py                    run every step
    python main.py --step data        only generate the dataset
    python main.py --step eda         only draw the fraud-trend charts
    python main.py --step train       only train and evaluate the models
    python main.py --rows 100000      generate a bigger dataset
    python main.py --fraud-rate 0.01  change the share of fraud
"""
import argparse
import time

import pandas as pd

from src import config, generate_data, train_models, visualize
from src.preprocessing import add_features

STEPS = ("data", "eda", "train")


def run_pipeline(steps=STEPS, n_rows=config.N_TRANSACTIONS, fraud_rate=config.FRAUD_RATE):
    started = time.perf_counter()
    config.ensure_dirs()

    if "data" in steps:
        print("\n[1/3] Generating transaction dataset ...")
        generate_data.save_datasets(n_rows, fraud_rate)

    if not config.TRANSACTIONS_CSV.exists():
        raise SystemExit("data/transactions.csv not found - run 'python main.py --step data' first.")
    df = add_features(pd.read_csv(config.TRANSACTIONS_CSV))

    if "eda" in steps:
        print("\n[2/3] Creating fraud-trend and transaction charts ...")
        visualize.create_eda_charts(df)

    if "train" in steps:
        print("\n[3/3] Training Random Forest and Logistic Regression ...")
        train_models.train_and_evaluate(df)

    print(f"\nDone in {time.perf_counter() - started:.0f}s.")
    print("Charts:  reports/figures/")
    print("Models:  models/")
    print("Next:    streamlit run app.py   (opens the dashboard in your browser)")


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection pipeline")
    parser.add_argument("--step", choices=STEPS, action="append",
                        help="run only this step (can be repeated); default: all steps")
    parser.add_argument("--rows", type=int, default=config.N_TRANSACTIONS,
                        help=f"transactions to generate (default {config.N_TRANSACTIONS:,})")
    parser.add_argument("--fraud-rate", type=float, default=config.FRAUD_RATE,
                        help=f"share of fraud transactions (default {config.FRAUD_RATE})")
    args = parser.parse_args()
    run_pipeline(args.step or STEPS, args.rows, args.fraud_rate)


if __name__ == "__main__":
    main()
