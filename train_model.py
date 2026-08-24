"""
train_model.py
----------------
Trains a Linear Regression model that predicts the average (modal) crop
price given:
    - State
    - District
    - Commodity (crop name)

The categorical inputs are one-hot encoded inside an sklearn Pipeline,
so the exact same preprocessing is automatically applied at prediction
time in the FastAPI service (no train/serve skew).

Run:
    python3 train_model.py
Produces:
    model.pkl          -> trained sklearn Pipeline (encoder + regressor)
    metadata.json       -> valid States / Districts / Commodities + metrics
"""

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

DATA_PATH = "raw_data.xlsx"
MODEL_PATH = "model.pkl"
METADATA_PATH = "metadata.json"

FEATURES = ["State", "District", "Commodity"]
TARGET = "Modal_x0020_Price"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={"Modal_x0020_Price": TARGET})
    # basic cleaning
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df[TARGET] > 0]  # drop bad/zero price rows
    for col in FEATURES:
        df[col] = df[col].astype(str).str.strip()
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURES,
            )
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return pipeline


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows after cleaning.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline -> {MODEL_PATH}")

    metadata = {
        "states": sorted(df["State"].unique().tolist()),
        "districts": sorted(df["District"].unique().tolist()),
        "commodities": sorted(df["Commodity"].unique().tolist()),
        "state_district_map": {
            state: sorted(df.loc[df["State"] == state, "District"].unique().tolist())
            for state in df["State"].unique()
        },
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "n_rows_trained": len(df),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
