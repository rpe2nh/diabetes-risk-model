from ucimlrepo import fetch_ucirepo
import pandas as pd

# Download the CDC Diabetes Health Indicators dataset (UCI id 891)
cdc = fetch_ucirepo(id=891)

# Combine the health indicators and the diabetes label into one table
df = pd.concat([cdc.data.features, cdc.data.targets], axis=1)
df.to_csv("data/diabetes_brfss2015.csv", index=False)

print("Rows, columns:", df.shape)
print(df.columns.tolist())
print(df.iloc[:, -1].value_counts(normalize=True))
