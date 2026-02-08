import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier

def run_model(df, anchor_date, usecase):
    X = df.drop(columns=["customer_id", "converted"]) 
    y= df["converted"]
    y_test = df["converted"]
    X_test = X

    if (not Path(f"output/xgb_{anchor_date}.pkl").exists()):
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


    # Save model
    joblib.dump(pipe, f"output/xgb_{anchor_date}.pkl")
    return y_test, y_pred, y_proba, pipe