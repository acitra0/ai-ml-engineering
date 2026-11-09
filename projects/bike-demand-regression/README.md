# Bike Sharing Demand Prediction — Regression Project

## 1. Project Overview

This project develops a machine learning model to predict the total number of daily bike rentals (`cnt`) using calendar information available one day before the prediction date.

The project demonstrates an end-to-end regression workflow, including:

- Exploratory feature engineering
- Chronological train/validation/test splitting
- Linear regression and Ridge regularization
- Hyperparameter tuning and model selection
- Model serialization using Joblib
- Independent evaluation on held-out test data
- Reproducible training and feature engineering

**Final selected model:** Scikit-learn `LinearRegression` with 14 engineered calendar features.

## 2. Dataset

**Source:** UCI Machine Learning Repository — Bike Sharing Dataset

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

**File:** `day.csv`

**Dataset characteristics:**

- 731 daily observations
- Date range: January 2011 – December 2012
- Target variable: `cnt` (total daily bike rentals)
- Prediction horizon: One day ahead

### Chronological Data Split

| Partition | Observations | Purpose |
|---|---:|---|
| Training | 511 | Fit candidate models |
| Validation | 109 | Compare models and hyperparameters |
| Test | 111 | Final held-out evaluation |
| Total | 731 | |

Data was kept in chronological order to simulate predicting future demand using past observations.

After selecting the final model, training and validation data were combined into 620 observations for the final training run.

The test set was not used to select features, tune hyperparameters, or choose the model.

## 3. Feature Engineering

The final model uses 14 engineered predictors.

### Numeric and Binary Features (5)

- `yr` — Year indicator (2011 or 2012)
- `holiday` — Whether the day is a holiday
- `workingday` — Whether the day is a working day
- `day_sin` — Sine transformation of the day of year
- `day_cos` — Cosine transformation of the day of year

Sine and cosine transformations encode annual seasonality as a cyclical pattern.

The encoding accounts for leap years by using either 365 or 366 days.

### Categorical Features (9)

**Weekday:** Six one-hot encoded indicators (`weekday_1` through `weekday_6`).

Sunday (`weekday_0`) is the reference category.

**Season:** Three one-hot encoded indicators (`season_2`, `season_3`, `season_4`).

Season 1 is the reference category.

Reference categories were excluded to avoid perfect multicollinearity with the intercept.

### Feature Availability and Leakage Prevention

Observed weather variables and rental-derived variables (`casual`, `registered`) were excluded.

Actual future weather measurements and rental counts would not be available when predicting tomorrow's demand.

The shared `build_features()` function also enforces consistent feature columns and order using pandas `.reindex()`, including when categorical values are absent from a small input batch.

## 4. Model Selection

Three model configurations were initially evaluated.

| Model | Configuration | Validation MAE | Validation RMSE |
|---|---|---:|---:|
| A | Basic linear regression | 1,124.78 | 1,300.45 |
| B | Engineered linear regression | 711.21 | 932.96 |
| C | Engineered Ridge (α=1.0) | 710.87 | 931.10 |

Model B substantially improved performance compared with Model A.

Additional Ridge experiments tested alpha values from 0.001 to 100.

The best observed Ridge validation MAE was approximately 710.76 at α=3.0.

However, this represented an improvement of only approximately 0.45 rentals compared with Model B.

### Final Decision

**Model B was selected** because its performance was nearly identical to the tuned Ridge model while avoiding an additional scaling and regularization configuration.

The marginal validation improvement from Ridge was insufficient evidence of a meaningful generalization advantage.

The selection was completed before evaluating the held-out test set.

## 5. Final Evaluation Results

After model selection, Model B was retrained using all 620 training and validation observations.

The final model was then evaluated on the remaining 111 observations.

| Metric | Validation | Held-Out Test |
|---|---:|---:|
| MAE | 711.21 | 1,081.37 |
| RMSE | 932.96 | 1,464.35 |

**Note:** Validation results were obtained from a model trained on 511 observations, while test results were obtained after refitting on 620 observations.

### Interpretation

The final test MAE was approximately 52% higher than the earlier validation MAE.

This indicates weaker performance on the later held-out period.

Possible explanations include changing demand patterns, omitted predictors, seasonal effects not fully captured by the model, and distribution shift.

The difference alone does not prove overfitting because the evaluation periods and training sample sizes differ.

The test results were retained without further model tuning.

## 6. Project Structure

```text
day-14-final-regression/
├── final_regression.ipynb
├── features.py
├── train.py
├── evaluate.py
├── test.py
├── model.joblib
└── README.md
```

**File responsibilities:**

- `final_regression.ipynb` — Hyperparameter tuning and model-selection experiments
- `features.py` — Shared feature-engineering function
- `train.py` — Dataset preparation, final model training, and serialization
- `evaluate.py` — Load saved model and calculate test metrics
- `test.py` — Development-time feature-engineering checks
- `model.joblib` — Serialized fitted LinearRegression model

The original dataset is stored at:

`zoomcamp/01-intro/1.9 Bike Sharing/data/day.csv`

## 7. Reproducing the Results

### Requirements

Python 3.11 and the following packages:

- pandas
- numpy
- scikit-learn
- joblib

Install dependencies in an activated virtual environment:

```bash
python -m pip install pandas numpy scikit-learn joblib
```

For exact environment reproducibility, package versions should also be pinned in a dependency file.

### Train the Model

From the repository root:

```bash
python zoomcamp/02-regression/day-14-final-regression/train.py
```

This loads the dataset, builds the engineered features, fits the final linear regression model on 620 observations, and saves `model.joblib`.

### Evaluate the Model

```bash
python zoomcamp/02-regression/day-14-final-regression/evaluate.py
```

Expected metrics, to two decimal places:

```text
Mean Absolute Error: 1081.37
Root Mean Squared Error: 1464.35
```

The evaluation script loads the existing fitted model without retraining it.

### Data Integrity and Validation

The project includes checks for:

- Dataset and partition dimensions
- Expected 14-feature input schema
- Missing values
- Matching feature and target lengths
- Chronological ordering
- Consistent training and evaluation feature names and order
- Feature generation when weekday or season categories are absent

## 8. Limitations

**1. Historical dataset**

The dataset contains observations only from 2011–2012. Its patterns may not generalize to modern bike-sharing demand without new data and retraining.

**2. Limited feature availability**

The model excludes weather conditions, which can materially affect bike rental demand. Future versions could incorporate weather forecasts available at prediction time.

**3. Dataset-specific year encoding**

The `yr` variable represents only 2011 and 2012. It is not suitable for extrapolating to later calendar years without redesign.

**4. Distribution shift**

The higher test error suggests that relationships learned from earlier observations may not generalize consistently across time.

**5. Limited model selection evidence**

Hyperparameters were evaluated against one validation period. Chronological cross-validation could provide more reliable estimates of model-selection stability.

**6. Linear model assumptions**

The model can represent additive relationships between engineered features and rental demand but may miss complex nonlinear interactions.

**7. Deployment readiness**

The project demonstrates reproducible offline training and evaluation, not a fully deployed forecasting service. Operational monitoring, retraining, robust input validation, and future-year feature handling remain necessary.

## 9. Future Improvements

- Introduce time-series cross-validation.
- Incorporate forecasted weather features without future-information leakage.
- Analyze prediction residuals by month and season.
- Evaluate nonlinear regression models.
- Add automated unit tests and dependency version pinning.
- Build a prediction interface and monitor performance over time.

## 10. Key Takeaways

This project demonstrates that feature engineering can produce greater improvements than switching to a more complex algorithm.

It also demonstrates the importance of separating model selection from final evaluation, maintaining consistent feature transformations, testing edge cases, and reporting held-out performance without selectively tuning toward the test results.

**Final outcome:** A reproducible regression workflow with a selected LinearRegression model, shared feature engineering, serialized model artifact, and independently reproduced held-out test metrics.