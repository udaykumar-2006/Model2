# Crop Price Prediction (Linear Regression + FastAPI)

Predicts the **average (modal) mandi price** of a crop given:
- **State**
- **District**
- **Commodity** (crop name)

## Files
| File | Purpose |
|---|---|
| `raw_data.xlsx` | Your original dataset |
| `train_model.py` | Cleans data, trains the Linear Regression pipeline, saves `model.pkl` + `metadata.json` |
| `model.pkl` | Trained sklearn Pipeline (OneHotEncoder + LinearRegression) |
| `metadata.json` | Valid states/districts/commodities + training metrics, used for input validation |
| `app.py` | FastAPI service that loads `model.pkl` and exposes prediction endpoints |
| `requirements.txt` | Python dependencies |

## How it works
Your dataset had `State`, `District`, `Commodity` as categorical text columns and
`Modal_Price` as the target. The pipeline:
1. **OneHotEncoder** turns State/District/Commodity into numeric columns (unseen values are ignored safely rather than crashing).
2. **LinearRegression** fits on those encoded features to predict `Modal_Price`.

On the held-out test set:
- **R² ≈ 0.80**
- **MAE ≈ ₹1,190**
- **RMSE ≈ ₹2,106**

Note: your data was a single-day snapshot (all rows dated 24/08/2026), so "avg price at that
time" here means the average modal price for that State + District + Commodity combination
as seen in the dataset — there isn't enough date history yet for real time-series trends. If you
later collect data across multiple dates, you can add `Arrival_Date` (e.g. month, day-of-year) as
an extra feature and retrain for time-based predictions.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train (already done, but to retrain)

```bash
python3 train_model.py
```

This regenerates `model.pkl` and `metadata.json` from `raw_data.xlsx`.

## 2. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open interactive docs at: **http://127.0.0.1:8000/docs**

## 3. Endpoints

### `POST /predict`
Request body:
```json
{
  "state": "Haryana",
  "district": "Sonipat",
  "commodity": "Tomato"
}
```
Response:
```json
{
  "state": "Haryana",
  "district": "Sonipat",
  "commodity": "Tomato",
  "predicted_avg_price": 1793.09,
  "unit": "INR per quintal"
}
```

### `GET /states`
List all valid states.

### `GET /districts/{state}`
List valid districts for a given state, e.g. `/districts/Haryana`.

### `GET /commodities`
List all valid crop/commodity names.

### `GET /metrics`
Returns the model's training metrics (MAE, RMSE, R²).

If you pass a state/district/commodity that wasn't in the training data, the API
returns a `422` error telling you which value is unrecognized, instead of silently
giving an unreliable prediction.

## Improving accuracy later
Linear Regression on one-hot categories gives a solid baseline (R² ≈ 0.80) but is
limited to purely additive effects. If you want higher accuracy down the line, consider:
- **Random Forest / Gradient Boosting (e.g. XGBoost)** — usually a big accuracy jump on this kind of categorical + price data.
- Adding `Market`, `Grade`, `Variety` as extra features.
- Adding `Arrival_Date` features once you have multi-date data, to capture seasonality.
