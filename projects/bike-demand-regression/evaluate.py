#Import Libraries
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
import joblib
from features import build_features

#Load the Model
script_dir = Path(__file__).resolve().parent

model_path = script_dir / "model.joblib"

modelB = joblib.load(model_path)

#Loading and sorting data
script_dir = Path(__file__).resolve().parent

data_path = script_dir.parent.parent / "zoomcamp" / "01-intro" / "1.9 Bike Sharing" / "data" / "day.csv"

df = pd.read_csv(data_path)

df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values("dteday").reset_index(drop=True)

X_modelB = build_features(df)

#Test Dataset
X_test = X_modelB.iloc[620:]
y_test = df["cnt"].iloc[620:]

print(X_test.shape, y_test.shape)

print(type(modelB))
print(modelB.n_features_in_)
print(modelB.feature_names_in_)

assert list(modelB.feature_names_in_) == list(X_test.columns), \
    "Feature names or order do not match training!"

#Test Model
modelB_pred = modelB.predict(X_test)

#Model Errors
modelB_mae = mean_absolute_error(y_test, modelB_pred)
modelB_rmse = root_mean_squared_error(y_test, modelB_pred)

print(f"Mean Absolute Error: {modelB_mae}, Root Mean Squared Error: {modelB_rmse}")
