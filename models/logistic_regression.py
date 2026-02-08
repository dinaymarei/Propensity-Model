import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np
from tabulate import tabulate
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

def run_model(df, anchor_date):

    # Define features and target
    X = df.drop(columns=["customer_id", "converted"])
    y = df["converted"]


    X_test = X
    y_test = y

    if not Path("output/lr_propensity_model.pkl").exists():
        # Create pipeline with XGBoost
        pipe = Pipeline([ ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Fit model
        pipe.fit(X_train, y_train)

        # Feature importance
        coef_df = pd.DataFrame(
        {"feature": X_train.columns, 
        "coeff": pipe.named_steps["model"].coef_[0]

        }
        )
        coef_df["odds_ratio"] = np.exp(coef_df["coeff"])

    # Export feature coeffecients
        coef_df.to_csv("importance/lr_feature_coef.csv")

    else:
        # Load cached model
        pipe = joblib.load(f"output/lr_{anchor_date}.pkl")
        print("Model loaded from cache")

    # Predictions
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Top-K Capture Rate
    comparison_df = pd.DataFrame({"actual": y_test.values, "prob": y_proba}).sort_values(by="prob", ascending=False)
    k = int(0.10 * len(comparison_df))
    top_k = comparison_df.head(k)
    capture_rate = top_k["actual"].sum() / comparison_df["actual"].sum()
    baseline_conversion = y_test.sum() / len(y_test)
    precision_at_k = top_k["actual"].mean()

    # Print metrics
    metrics = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Anchor Date": anchor_date,
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

    if Path(f"results/results.csv").exists():
        results_df = pd.read_csv("results/results.csv")
    else:
        results_df = pd.DataFrame(columns=["Model", "Anchor Date", "Accuracy", "AUC", "Precision", "Recall", "Capture Rate (Recall @K =10%)", "Baseline Conversion Rate", "Precision at K"])
    results_df = pd.concat([results_df, metrics], ignore_index=True)
    results_df.to_csv("results/results.csv", index=False)

    print(tabulate(metrics, tablefmt="pretty", showindex=False, headers="keys"))

    # Save pipeline
    joblib.dump(pipe, f"output/lr_{anchor_date}.pkl")
