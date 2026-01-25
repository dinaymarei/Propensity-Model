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

# Render input data
ANCHOR_DATE = '2025-11-01'
USE_CASE = "food"



df = render(anchor_date=ANCHOR_DATE,use_case=USE_CASE)
print(df)

# Create X and Y 
X = df.drop(columns=["customer_id", "converted"])
y= df["converted"]

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

# Export feature coeffecients
coef_df.to_csv("results/feature_coef_{{USE_CASE}}_{{ANCHOR_DATE}}.csv")

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


metrics = pd.DataFrame( [
    {"Accuracy": acc, 
     "AUC": auc, 
     "Precision": precision, 
     "Recall": recall, 
     "Capture Rate (Recall @K =10%)": capture_rate
     }
]
)
print(tabulate(metrics, tablefmt="pretty", showindex=False, headers="keys"))



