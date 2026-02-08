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

    X = df.drop(columns=["customer_id", "converted"])
    y = df["converted"]

    X_test = X
    y_test = y

    if not Path("output/lr_propensity_model.pkl").exists():
        # Create pipeline with XGBoost
        pipe = Pipeline([ ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

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
        pipe = joblib.load(f"output/lr_{anchor_date}.pkl")
        print("Model loaded from cache")

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Save model
    joblib.dump(pipe, f"output/lr_{anchor_date}.pkl")

    #Calculate Scores
    return y_test, y_pred, y_proba, pipe

