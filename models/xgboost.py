import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from google.cloud import bigquery
from configs.lookup import BQ_PROJECT, BQ_DATASET


def _write_to_bq(df: pd.DataFrame, table: str):
    client = bigquery.Client(project=BQ_PROJECT)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df, f"{BQ_DATASET}.{table}", job_config=job_config).result()


def run_model(df: pd.DataFrame, anchor_date: str, mode: str = "train"):
    drop_cols = ["customer_id", "converted", "primary_fp"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model_path = Path(f"output/xgb_{anchor_date}.pkl")

    if mode == "train" or not model_path.exists():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                n_jobs=-1,
                random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)

        importance_df = pd.DataFrame({
            "anchor_date": anchor_date,
            "feature": X_train.columns,
            "importance": pipe.named_steps["model"].feature_importances_,
        })
        _write_to_bq(importance_df, "xgb_feature_importance")

        joblib.dump(pipe, model_path)

    else:
        pipe = joblib.load(model_path)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    predictions = pd.DataFrame({
        "anchor_date": anchor_date,
        "customer_id": df.loc[X_test.index, "customer_id"],
        "actual_converted": y_test.values,
        "predicted_converted": y_pred,
        "predicted_probability": y_proba,
    })
    _write_to_bq(predictions, "xgb_predictions")

    return y_test, y_pred, y_proba, pipe
