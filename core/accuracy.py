from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from google.cloud import bigquery
from configs.lookup import BQ_PROJECT, BQ_DATASET


def score(model: str, y_test, y_pred, y_proba, anchor_date: str, use_case: str, fps: list = None):
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    comparison_df = pd.DataFrame({"actual": y_test.values, "prob": y_proba}).sort_values("prob", ascending=False)
    k = int(0.10 * len(comparison_df))
    top_k = comparison_df.head(k)

    capture_rate = top_k["actual"].sum() / comparison_df["actual"].sum()
    baseline_conversion = y_test.sum() / len(y_test)
    precision_at_k = top_k["actual"].mean()

    metrics = {
        "run_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "anchor_date": anchor_date,
        "use_case": use_case,
        "filtered_fps": ", ".join(fps) if fps else "All",
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "capture_rate_at_10pct": round(capture_rate, 4),
        "baseline_conversion_rate": round(baseline_conversion, 4),
        "precision_at_k": round(precision_at_k, 4),
    }

    print(tabulate([metrics], tablefmt="pretty", showindex=False, headers="keys"))

    client = bigquery.Client(project=BQ_PROJECT)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(
        pd.DataFrame([metrics]),
        f"{BQ_DATASET}.model_results",
        job_config=job_config,
    ).result()
