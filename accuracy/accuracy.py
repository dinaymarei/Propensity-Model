from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import pandas as pd 
from tabulate import tabulate
from pathlib import Path

def score(model, y_test, y_pred, y_proba, anchor_date, use_case, fps=None):

    # Core Accuracy Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Auxilary Propensity-specific Metrics
    comparison_df = pd.DataFrame({"actual": y_test.values, "prob": y_proba}).sort_values(by="prob", ascending=False)
    k = int(0.10 * len(comparison_df))

    top_k = comparison_df.head(k)
    capture_rate = top_k["actual"].sum() / comparison_df["actual"].sum()
    baseline_conversion = y_test.sum() / len(y_test)
    precision_at_k = top_k["actual"].mean()
 
    metrics = pd.DataFrame([
        {
            "Model": model,
            "Anchor Date": anchor_date,
            "Use Case": use_case,
            "Filtered FPs": ", ".join(fps) if fps else "All",
            "Accuracy": acc,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "Capture Rate (Recall @K =10%)": capture_rate,
            "Baseline Conversion Rate": baseline_conversion,
            "Precision at K": precision_at_k
        }
    ])
    print(tabulate(metrics, tablefmt="pretty", showindex=False, headers="keys"))

    # Save results and configuration 
    if Path(f"results/results.csv").exists():
        results_df = pd.read_csv("results/results.csv")
    else:
        results_df = pd.DataFrame(columns=["Model", "Anchor Date", "Use Case","Filtered FPs", "Accuracy", "AUC", "Precision",
                                            "Recall", "Capture Rate (Recall @K =10%)", "Baseline Conversion Rate",
                                            "Precision at K"])

    results_df = pd.concat([results_df, metrics], ignore_index=True)
    results_df.to_csv("results/results.csv", index=False)   