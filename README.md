# IPO Listing Gain Predictor

Predicts whether an upcoming Indian IPO will list at a **premium or discount**, and
estimates the likely **listing-day gain %**, based on subscription demand
(QIB/HNI/RII categories) and issue details. Built on 16 years (2010–2026) of real
Indian IPO data, served as a FastAPI service with prediction logging to a database.

## Why this project

IPO subscription data is published days before listing — institutional (QIB) and
high-net-worth (HNI) demand is known well before retail investors get access to
listing-day prices. This project turns that public data into a predictive signal,
the same way analysts/desks informally do today, but as a reproducible model.

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Classification (premium vs discount) | Random Forest | AUC | **0.81** |
| Classification (premium vs discount) | Random Forest | Accuracy | **75.4%** |
| Regression (exact listing gain %) | Linear Regression | R² | **0.40** |
| Regression (exact listing gain %) | Linear Regression | MAE | **13.2 pp** |

### A finding worth calling out

Initial models were evaluated with a time-based train/test split (train on
2010–2024, test on 2025–2026, simulating real deployment). The classifier held up
well, but the regressor's R² went **negative** — worse than predicting the average.

Digging in: IPO listing gains cooled sharply in the most recent period (mean gain
dropped from ~18% in 2010–2024 to ~9% in late 2024–2026 — a market regime shift).
A regressor trained on the hotter era systematically **overpredicts magnitude** in
the cooler one. The classifier was unaffected because **direction** (will it list
up or down) is a more stable signal than **magnitude** (by how much) across market
regimes.

This is why the app leads with the classifier's premium/discount call and treats
the regressor's exact % as a secondary, lower-confidence estimate — and why
`/model-info` documents this explicitly instead of hiding it.

## What drives the prediction

Feature importance (Random Forest classifier) confirms market intuition:
**overall subscription demand** matters most, followed by **institutional (HNI/QIB)
demand** — retail (RII) subscription is the weakest individual signal, consistent
with institutional investors having better information access.

## Tech stack

- **Data**: pandas, feature engineering (log transforms, derived ratios)
- **Models**: scikit-learn (Random Forest, Gradient Boosting, Linear/Logistic baselines)
- **API**: FastAPI + Pydantic validation
- **Database**: SQLite + SQLAlchemy (logs every prediction made)
- **Deployment**: Render / Railway (see below)

## Project structure

```
ipo_project/
├── data/
│   ├── ipo_raw.xlsx          # raw Kaggle dataset (652 IPOs, 2010-2026)
│   └── ipo_clean.csv         # cleaned + feature-engineered
├── notebooks/
│   ├── 01_clean_and_features.py
│   └── 02_train_models.py
├── models/                   # saved .pkl models + metadata
├── api/
│   ├── main.py                # FastAPI app
│   └── database.py            # SQLAlchemy models
├── requirements.txt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI.

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "issue_size_crores": 1200,
    "qib_subscription": 15.5,
    "hni_subscription": 8.2,
    "rii_subscription": 3.1,
    "total_subscription": 10.4,
    "offer_price": 250,
    "year": 2026
  }'
```

### Example response

```json
{
  "predicted_listing_gain_pct": 1.67,
  "listed_at_premium": true,
  "premium_probability": 0.601,
  "confidence_note": "Moderate confidence — subscription signals are mixed"
}
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Predict listing gain % and premium/discount |
| GET | `/predictions` | View recent logged predictions |
| GET | `/model-info` | Model metadata, scores, and known limitations |
| GET | `/health` | Health check |

## Data source

[IPO Data India 2010–2026, Kaggle](https://www.kaggle.com/datasets/karanammithul/ipo-data-india-2010-2025)
— 652 IPOs with offer price, subscription multiples (QIB/HNI/RII/Total), listing
price, and current market price.

## Limitations & next steps

- 648 usable rows after cleaning — enough for classical ML, not deep learning
- No macro features (Nifty/Sensex trend, sector, grey market premium) — adding
  these would likely close some of the regressor's gap
- Regime shift suggests a **rolling retrain** strategy would outperform a static
  model in production
