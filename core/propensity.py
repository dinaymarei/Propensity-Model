import argparse
import os
from .render import render
from accuracy.accuracy import score

def main(anchor_date: str, model: str, use_case: str):

    if use_case == "food": # Filter on specific FPs where food is enabled
        fps = ["Al Rehab City FP #1", "Maadi FP #1"]
    else: 
        fps = None
    df = render(anchor_date=anchor_date, use_case=use_case, fps = fps)

    if model == "logistic":
        from models.logistic_regression import run_model
    elif model == "xgboost":
        from models.xgboost import run_model
    else:
        raise ValueError("Model must be 'logistic' or 'xgboost'")

    y_test, y_pred, y_proba, pipe = run_model(df, anchor_date)
    score(model, y_test, y_pred, y_proba, anchor_date, use_case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--anchor-date", required=True)
    parser.add_argument("--model", required=True, choices=["logistic", "xgboost"])
    parser.add_argument("--use-case", default="beauty")

    args = parser.parse_args()

    main(args.anchor_date, args.model, args.use_case)
