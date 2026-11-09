from features import build_features
from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path(__file__).resolve().parent

data_path = script_dir.parent.parent / "01-intro" / "1.9 Bike Sharing" / "data" / "day.csv"

df = pd.read_csv(data_path)

test_input = df.drop(columns=["cnt"])

X = build_features(test_input)

assert X.shape == (731, 14)
assert "cnt" not in X.columns

sample = df.loc[
    (df["season"] == 2) & (df["weekday"] == 1)
].head(3).drop(columns=["cnt"])

X_sample = build_features(sample)

print(X_sample.shape)
print(X_sample.columns.tolist())