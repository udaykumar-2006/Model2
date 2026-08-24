"""
app.py
-------
FastAPI service that serves the Linear Regression crop-price model.

Run locally:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Docs (Swagger UI) once running:
    http://127.0.0.1:8000/docs
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Load model + metadata once at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

with open(METADATA_PATH) as f:
    metadata = json.load(f)

VALID_STATES = set(metadata["states"])
VALID_COMMODITIES = set(metadata["commodities"])
STATE_DISTRICT_MAP = metadata["state_district_map"]

app = FastAPI(
    title="Crop Price Prediction API",
    description="Predicts the average (modal) market price of a crop "
    "given its State, District, and Commodity name, using a "
    "Linear Regression model trained on mandi price data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    state: str = Field(..., example="Haryana", description="State name")
    district: str = Field(..., example="Sonipat", description="District name")
    commodity: str = Field(..., example="Tomato", description="Crop / commodity name")


class PredictionResponse(BaseModel):
    predicted_avg_price: float
    unit: str = "INR per quintal"


# ---------------------------------------------------------------------------



@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    state = payload.state.strip()
    district = payload.district.strip()
    commodity = payload.commodity.strip()

    # Soft validation: warn-style checks against known categories.
    # (Model still works on unseen values thanks to handle_unknown="ignore",
    #  but the prediction will fall back to the intercept-only estimate,
    #  so we flag it clearly instead of pretending it's precise.)
    unknown_parts = []
    if state not in VALID_STATES:
        unknown_parts.append(f"state '{state}'")
    if commodity not in VALID_COMMODITIES:
        unknown_parts.append(f"commodity '{commodity}'")
    if state in STATE_DISTRICT_MAP and district not in STATE_DISTRICT_MAP[state]:
        unknown_parts.append(f"district '{district}' (for state '{state}')")

    if unknown_parts:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unknown value(s) not seen during training: "
                + ", ".join(unknown_parts)
                + ". Use /states, /districts/{state}, /commodities to see valid options."
            ),
        )

    input_df = pd.DataFrame(
        [{"State": state, "District": district, "Commodity": commodity}]
    )
    prediction = float(model.predict(input_df)[0])
    prediction = max(prediction, 0.0)  # price can't be negative

    return PredictionResponse(
        predicted_avg_price=round(prediction, 2),
    )
