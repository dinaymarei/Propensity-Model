import argparse
import os
from .render import render


def main(anchor_date: str, model: str, use_case: str):

    input_folder = f"input/{anchor_date}.csv"
    print(f"Using cached data from: {input_folder}")

    df = render(anchor_date=anchor_date, use_case=use_case)

    if model == "logistic":
        from models.logistic_regression import run_model
    elif model == "xgboost":
        from models.xgboost import run_model
    else:
        raise ValueError("Model must be 'logistic' or 'xgboost'")

    run_model(df, anchor_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--anchor-date", required=True)
    parser.add_argument("--model", required=True, choices=["logistic", "xgboost"])
    parser.add_argument("--use-case", default="beauty")

    args = parser.parse_args()

    main(args.anchor_date, args.model, args.use_case)
