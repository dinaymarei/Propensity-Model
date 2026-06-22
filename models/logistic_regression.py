import io
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from tabulate import tabulate
from configs.lookup import BQ_PROJECT, BQ_DATASET, DROP_COLS

logger = logging.getLogger(__name__)


def _write_to_bq(df: pd.DataFrame, table: str, append: bool = False):
    from google.cloud import bigquery
    client = bigquery.Client(project=BQ_PROJECT)
    disposition = "WRITE_APPEND" if append else "WRITE_TRUNCATE"
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    job_config = bigquery.LoadJobConfig(
        write_disposition=disposition,
        source_format=bigquery.SourceFormat.PARQUET,
    )
    client.load_table_from_file(buffer, f"{BQ_DATASET}.{table}", job_config=job_config).result()
    logger.info("Written to BQ table: %s.%s", BQ_DATASET, table)


def _model_path(use_case: str) -> str:
    return f"models/{use_case}_logistic.pkl"


def train(df: pd.DataFrame, anchor_date: str, use_case: str):
    X = df.drop(columns=df.columns.intersection(DROP_COLS))
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, _model_path(use_case))
    logger.info("Model saved to %s", _model_path(use_case))

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    propensity_df = pd.DataFrame({
        "anchor_date": anchor_date,
        "customer_id": df["customer_id"],
        "propensity_score": pipe.predict_proba(X)[:, 1],
        "converted": df["converted"].values,
    }).sort_values("propensity_score", ascending=False)
    _write_to_bq(propensity_df, f"{use_case}_propensity_scores", append=True)

    return y_test, y_pred, y_proba


def score(df: pd.DataFrame, anchor_date: str, use_case: str):
    pipe = joblib.load(_model_path(use_case))
    logger.info("Loaded model from %s", _model_path(use_case))

    X = df.drop(columns=df.columns.intersection(DROP_COLS))

    propensity_df = pd.DataFrame({
        "anchor_date": anchor_date,
        "customer_id": df["customer_id"],
        "propensity_score": pipe.predict_proba(X)[:, 1],
    }).sort_values("propensity_score", ascending=False)
    _write_to_bq(propensity_df, f"{use_case}_propensity_scores", append=True)
    logger.info("Scored %d customers", len(propensity_df))


def analyze(df: pd.DataFrame, anchor_date: str, use_case: str):
    pipe = joblib.load(_model_path(use_case))
    logger.info("Loaded model from %s", _model_path(use_case))

    X = df.drop(columns=df.columns.intersection(DROP_COLS))
    y = df["converted"]

    # Coefficients sorted by absolute value
    coef_df = pd.DataFrame({
        "anchor_date": anchor_date,
        "feature": X.columns,
        "coefficient": pipe.named_steps["model"].coef_[0],
    })
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    coef_df = coef_df.reindex(
        coef_df["coefficient"].abs().sort_values(ascending=False).index
    )
    _write_to_bq(coef_df, f"{use_case}_feature_coef", append=True)

    print("\n--- Feature Coefficients ---")
    print(tabulate(coef_df.drop(columns="anchor_date"), headers="keys", tablefmt="pretty", showindex=False, floatfmt=".4f"))

    # Permutation importance
    logger.info("Running permutation importance (n_repeats=20)...")
    result = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=20, random_state=42)
    perm_df = pd.DataFrame({
        "anchor_date": anchor_date,
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    _write_to_bq(perm_df, f"{use_case}_permutation_importance", append=True)

    print("\n--- Permutation Importance ---")
    print(tabulate(perm_df.drop(columns="anchor_date"), headers="keys", tablefmt="pretty", showindex=False, floatfmt=".4f"))
