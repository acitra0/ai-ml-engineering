#Import Libraries
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
import joblib
from features import build_features

#Loading and sorting data
script_dir = Path(__file__).resolve().parent

data_path = script_dir.parent.parent / "zoomcamp" / "01-intro" / "1.9 Bike Sharing" / "data" / "day.csv"

df = pd.read_csv(data_path)

df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values("dteday").reset_index(drop=True)

X_modelB = build_features(df)

assert df["dteday"].is_monotonic_increasing
assert X_modelB.index.equals(df.index)

#Full Training Dataset
X_full_train = X_modelB.iloc[:620]
y_full_train = df["cnt"].iloc[:620]

#Test Dataset
X_test = X_modelB.iloc[620:]
y_test = df["cnt"].iloc[620:]

#Assertion Tests
assert X_modelB.shape[0] == 731
assert X_full_train.shape[0] == 620
assert X_test.shape[0] == 111
assert X_modelB.shape[1] == 14
assert X_modelB.isna().sum().sum() == 0
assert len(X_full_train) == len(y_full_train)
assert len(X_test) == len(y_test)
assert df["dteday"].iloc[619] < df["dteday"].iloc[620]

#Train Model
modelB = LinearRegression()
modelB.fit(X_full_train, y_full_train)
model_path = script_dir / "model.joblib"

joblib.dump(modelB, model_path)

print(f"Model saved to: {model_path}")