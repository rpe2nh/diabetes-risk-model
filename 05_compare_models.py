import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns="Diabetes_binary"), train["Diabetes_binary"]
X_test, y_test = test.drop(columns="Diabetes_binary"), test["Diabetes_binary"]

# Model 1: logistic regression (same as Step 4)
logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000)
).fit(X_train, y_train)
print("Check - logistic regression weights are all valid numbers:",
      np.isfinite(logreg[-1].coef_).all())

# Model 2: gradient boosting (builds many small decision trees, each fixing the last one's mistakes)
gboost = HistGradientBoostingClassifier(
    class_weight="balanced", max_iter=300, random_state=42
).fit(X_train, y_train)

print("\nROC-AUC on the test set:")
for name, model in [("Logistic regression", logreg), ("Gradient boosting", gboost)]:
    proba = model.predict_proba(X_test)[:, 1]
    print(f"  {name}: {roc_auc_score(y_test, proba):.3f}")

# How the decision threshold changes who gets flagged
proba = gboost.predict_proba(X_test)[:, 1]
print("\nGradient boosting at different thresholds:")
print("threshold | recall | precision | people flagged")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (proba >= t).astype(int)
    print(f"   {t:.1f}    | {recall_score(y_test, preds):.3f}  |   {precision_score(y_test, preds):.3f}   | {preds.sum():,}")