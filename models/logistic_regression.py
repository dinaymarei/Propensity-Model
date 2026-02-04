import pandas as pd 
from render_query import render
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np
from tabulate import tabulate
import joblib
from pathlib import Path

# Render input data
ANCHOR_DATE = '2025-12-01'
USE_CASE = "beauty"


df = render(anchor_date=ANCHOR_DATE,use_case=USE_CASE)
X = df.drop(columns=["customer_id", "converted", "is_baby_parent", "is_pet_owner", 
                        "office_ratio", "months_since_acquisition", "health_and_wellness_ratio"])

y= df["converted"]
y_test = df["converted"]
X_test = X

if (not Path("output/lr_propensity_model.pkl").exists()):
# Create Pipeline and Save Model
    pipe = Pipeline([ ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])



    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipe.fit(X_train,y_train)

    # Feature relevance
    coef_df = pd.DataFrame(
        {"feature": X_train.columns, 
        "coeff": pipe.named_steps["model"].coef_[0]

        }
    )
    coef_df["odds_ratio"] = np.exp(coef_df["coeff"])

    # Export feature coeffecients
    coef_df.to_csv("results/lr_feature_coef.csv")

else: 
 # Load model
    pipe = joblib.load("output/lr_propensity_model.pkl")
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
    {"Accuracy": acc, 
     "AUC": auc, 
     "Precision": precision, 
     "Recall": recall, 
     "Capture Rate (Recall @K =10%)": capture_rate,
     "Baseline Conversion Rate": baseline_conversion,
     "Precision at K": precision_at_k
    }
]
)
print(tabulate(metrics, tablefmt="pretty", showindex=False, headers="keys"))
joblib.dump(pipe, "output/lr_propensity_model.pkl")



