import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diabetes_brfss2015.csv")

# 1. Data quality checks
print("Missing values per column:")
print(df.isna().sum().sum(), "total missing")

print("\nExact duplicate rows:", df.duplicated().sum())

print("\nValue ranges:")
print(df.describe().T[["min", "max"]])

# 2. Split into training (80%) and test (20%) sets
X = df.drop(columns="Diabetes_binary")
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTrain size:", X_train.shape, "| diabetes rate:", round(y_train.mean(), 3))
print("Test size: ", X_test.shape, "| diabetes rate:", round(y_test.mean(), 3))

X_train.assign(Diabetes_binary=y_train).to_csv("data/train.csv", index=False)
X_test.assign(Diabetes_binary=y_test).to_csv("data/test.csv", index=False)