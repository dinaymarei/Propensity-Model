import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np
from tabulate import tabulate
import joblib
from pathlib import Path
from xgboost import XGBClassifier

def run_model(df, anchor_date):
    X = df.drop(columns=["customer_id", "converted"]) 
    y= df["converted"]
    y_test = df["converted"]
    X_test = X

    if (not Path(f"output/{anchor_date}.pkl").exists()):
        # Create Pipeline and Save Model
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
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        pipe.fit(X_train,y_train)

        # Feature importance
        importance = pipe.named_steps["model"].feature_importances_
        coef_df = pd.DataFrame({"feature": X_train.columns, "importance": importance})
        coef_df.to_csv(f"importance/xgb_{anchor_date}.csv", index=False)

    else: 
    # Load model
        pipe = joblib.load(f"output/{anchor_date}.pkl")
        print("Model loaded from cache")


    # Prediction Accuracy 
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]


    # Calculate Scores 
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    comparison_df = pd.DataFrame({"actual": y_test.values, "prob": y_proba}).sort_values(by="prob", ascending=False)
    k = int(0.10*len(comparison_df))
    top_k= comparison_df.head(k)
    capture_rate = top_k["actual"].sum()/ comparison_df["actual"].sum()
    baseline_conversion = y_test.sum()/len(y_test)

    precision_at_k = top_k["actual"].mean()

    metrics = pd.DataFrame( [

        { "Model": "XGBoost",
         "Anchor Date": anchor_date,
        "Accuracy": acc, 
        "AUC": auc, 
        "Precision": precision, 
        "Recall": recall, 
        "Capture Rate (Recall @K =10%)": capture_rate,
        "Baseline Conversion Rate": baseline_conversion, 
        "Precision at K": precision_at_k
        }
    ]
    )

    if Path(f"results/results.csv").exists():
        results_df = pd.read_csv("results/results.csv")
    else:
        results_df = pd.DataFrame(columns=["Model", "Anchor Date", "Accuracy", "AUC", "Precision", "Recall", "Capture Rate (Recall @K =10%)", "Baseline Conversion Rate", "Precision at K"])
    results_df = pd.concat([results_df, metrics], ignore_index=True)
    results_df.to_csv("results/results.csv", index=False)

    
    print(tabulate(metrics, tablefmt="pretty", showindex=False, headers="keys"))
    joblib.dump(pipe, f"output/xgboost_{anchor_date}.pkl")


