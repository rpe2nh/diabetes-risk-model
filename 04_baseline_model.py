import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns="Diabetes_binary"), train["Diabetes_binary"]
X_test, y_test = test.drop(columns="Diabetes_binary"), test["Diabetes_binary"]

def evaluate(name, model):
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy: ", round(accuracy_score(y_test, preds), 3))
    print("Recall:   ", round(recall_score(y_test, preds), 3), " <- share of diabetic people the model catches")
    print("Precision:", round(precision_score(y_test, preds, zero_division=0), 3), " <- share of flagged people who really have it")
    if hasattr(model, "predict_proba"):
        print("ROC-AUC:  ", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3))
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print(f"Caught {tp:,} of {tp + fn:,} diabetic people | {fp:,} false alarms")

# 1. "Dumb" baseline: always predicts no diabetes
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
evaluate("Always predict 'no diabetes'", dummy)

# 2. Logistic regression, weighted so the rarer diabetic cases count more
logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000)
).fit(X_train, y_train)
evaluate("Logistic regression", logreg)