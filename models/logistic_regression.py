import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from google.cloud import bigquery
from configs.lookup import BQ_PROJECT, BQ_DATASET


def _write_to_bq(df: pd.DataFrame, table: str):
    client = bigquery.Client(project=BQ_PROJECT)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df, f"{BQ_DATASET}.{table}", job_config=job_config).result()


def run_model(df: pd.DataFrame, anchor_date: str, mode: str = "train"):
    drop_cols = ["customer_id", "converted", "unique_views_in_cat"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    if mode == "train":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipe.fit(X_train, y_train)

        coef_df = pd.DataFrame({
            "anchor_date": anchor_date,
            "feature": X_train.columns,
            "coeff": pipe.named_steps["model"].coef_[0],
        })
        coef_df["odds_ratio"] = np.exp(coef_df["coeff"])
        _write_to_bq(coef_df, "lr_feature_coef")

        result = permutation_importance(
            pipe, X_test, y_test,
            scoring="roc_auc", n_repeats=20, random_state=42,
        )
        perm_df = pd.DataFrame({
            "anchor_date": anchor_date,
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        _write_to_bq(perm_df, "lr_permutation_importance")

        joblib.dump(pipe, "output/logistic.pkl")

    else:
        pipe = joblib.load("output/logistic.pkl")

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    predictions = pd.DataFrame({
        "anchor_date": anchor_date,
        "customer_id": df.loc[X_test.index, "customer_id"],
        "actual_converted": y_test.values,
        "predicted_converted": y_pred,
        "predicted_probability": y_proba,
    })
    _write_to_bq(predictions, "lr_predictions")

    # Score all customers for ranking output
    df = df.copy()
    df["score"] = pipe.predict_proba(X)[:, 1]
    df = df.sort_values("score", ascending=False)
    print(df.groupby(pd.qcut(df["score"], 10))["converted"].mean())

    return y_test, y_pred, y_proba, pipe
