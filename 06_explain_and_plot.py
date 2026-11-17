import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # harmless Apple math-library warnings (verified in Step 5)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import RocCurveDisplay, recall_score, precision_score

THRESHOLD = 0.5  # your chosen cutoff - change to 0.4 if you prefer catching more cases

os.makedirs("figures", exist_ok=True)
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns="Diabetes_binary"), train["Diabetes_binary"]
X_test, y_test = test.drop(columns="Diabetes_binary"), test["Diabetes_binary"]

logreg = make_pipeline(StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X_train, y_train)
gboost = HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                        random_state=42).fit(X_train, y_train)

# 1. ROC curves: how well each model ranks people by risk
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(logreg, X_test, y_test, name="Logistic regression", ax=ax)
RocCurveDisplay.from_estimator(gboost, X_test, y_test, name="Gradient boosting", ax=ax)
ax.plot([0, 1], [0, 1], "--", color="gray", label="Coin flip (AUC = 0.5)")
ax.set_title("ROC Curves on Held-Out Test Set")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("figures/roc_curves.png", dpi=150)

# 2. Which factors the model relies on most (permutation importance)
sample = X_test.sample(10000, random_state=42)
imp = permutation_importance(gboost, sample, y_test.loc[sample.index],
                             scoring="roc_auc", n_repeats=5, random_state=42)
importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
print("Top 10 factors (drop in ROC-AUC when that column is scrambled):")
print(importance.sort_values(ascending=False).head(10).round(4))

fig, ax = plt.subplots(figsize=(7, 5))
importance.tail(10).plot.barh(ax=ax, color="#232D4B")
ax.set_xlabel("Drop in ROC-AUC when the factor is scrambled")
ax.set_title("Top 10 Predictors of Diabetes Risk")
fig.tight_layout()
fig.savefig("figures/feature_importance.png", dpi=150)

# 3. Direction of each factor (logistic regression weights; positive = higher risk)
coefs = pd.Series(logreg[-1].coef_[0], index=X_train.columns).sort_values(ascending=False)
print("\nLogistic regression weights (positive = raises risk, negative = lowers it):")
print(coefs.round(3))

# 4. Threshold trade-off chart
proba = gboost.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.1, 0.9, 81)
recalls = [recall_score(y_test, proba >= t) for t in thresholds]
precisions = [precision_score(y_test, proba >= t, zero_division=0) for t in thresholds]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thresholds, recalls, label="Recall (share of cases caught)")
ax.plot(thresholds, precisions, label="Precision (share of flags that are correct)")
ax.axvline(THRESHOLD, color="gray", linestyle="--", label=f"Chosen threshold ({THRESHOLD})")
ax.set_xlabel("Decision threshold")
ax.set_title("Recall vs. Precision Trade-off (Gradient Boosting)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/threshold_tradeoff.png", dpi=150)

# 5. Final numbers at your chosen threshold
preds = proba >= THRESHOLD
print(f"\nAt threshold {THRESHOLD}: recall {recall_score(y_test, preds):.1%}, "
      f"precision {precision_score(y_test, preds):.1%}, "
      f"flags {preds.mean():.0%} of people")
print("\nCharts saved to the figures/ folder.")