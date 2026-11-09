def build_features(df):
    import numpy as np
    import pandas as pd
    df["day_of_year"] = df["dteday"].dt.day_of_year

    #One-Hot Encoding For Weekday and Season
    weekday_encoded = pd.get_dummies(df["weekday"], prefix="weekday", dtype=int).reindex(
    columns=[f"weekday_{i}" for i in range(1, 7)],
    fill_value=0
)
    season_encoded = pd.get_dummies(df["season"], prefix="season", dtype=int).reindex(
    columns=[f"season_{i}" for i in range(2, 5)],
    fill_value=0
)

    #Day of Year
    df["days_in_year"] = np.where(df['dteday'].dt.is_leap_year, 366, 365)
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / df["days_in_year"]
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / df["days_in_year"]
    )

    #Preparing Data
    X_engineered = df[["yr", "holiday", "workingday", "weekday", "season", "day_sin", "day_cos"]].drop(columns=["weekday", "season"])
    X_modelB = pd.concat([season_encoded, weekday_encoded, X_engineered], axis=1)

    return X_modelB