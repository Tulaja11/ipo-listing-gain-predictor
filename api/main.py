"""
IPO Listing Gain Predictor — FastAPI service.

Exposes:
  POST /predict        -> predicts listing gain % (regression) AND
                           premium/discount probability (classification)
  GET  /predictions     -> list past predictions logged to the database
  GET  /health          -> simple healthcheck
  GET  /model-info      -> model metadata (which algorithm, scores, features)

Run locally:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from datetime import datetime

from database import SessionLocal, init_db, PredictionLog

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="IPO Listing Gain Predictor",
    description="Predicts whether an upcoming IPO will list at a premium or "
                "discount, and estimates the likely listing-day gain %, "
                "based on subscription demand (QIB/HNI/RII) and issue details.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models once at startup ---
regressor = joblib.load(os.path.join(MODEL_DIR, "regressor.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
scaler_reg = joblib.load(os.path.join(MODEL_DIR, "scaler_reg.pkl"))
scaler_clf = joblib.load(os.path.join(MODEL_DIR, "scaler_clf.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

init_db()


class IPOInput(BaseModel):
    issue_size_crores: float = Field(..., gt=0, description="Total IPO issue size in INR crores")
    qib_subscription: float = Field(..., ge=0, description="QIB (institutional) subscription multiple, e.g. 12.5 means 12.5x")
    hni_subscription: float = Field(..., ge=0, description="HNI subscription multiple")
    rii_subscription: float = Field(..., ge=0, description="Retail (RII) subscription multiple")
    total_subscription: float = Field(..., ge=0, description="Overall subscription multiple across all categories")
    offer_price: float = Field(..., gt=0, description="IPO offer price per share (INR)")
    year: int = Field(default=datetime.now().year, description="Listing year, defaults to current year")

    class Config:
        json_schema_extra = {
            "example": {
                "issue_size_crores": 1200.0,
                "qib_subscription": 15.5,
                "hni_subscription": 8.2,
                "rii_subscription": 3.1,
                "total_subscription": 10.4,
                "offer_price": 250,
                "year": 2026,
            }
        }


class PredictionResponse(BaseModel):
    predicted_listing_gain_pct: float
    listed_at_premium: bool
    premium_probability: float
    confidence_note: str


import pandas as pd


def build_features(ipo: IPOInput) -> "pd.DataFrame":
    log_issue_size = np.log1p(ipo.issue_size_crores)
    rii_to_qib = ipo.rii_subscription / ipo.qib_subscription if ipo.qib_subscription > 0 else 0
    is_oversubscribed = 1 if ipo.total_subscription > 1 else 0
    qib_dominance = ipo.qib_subscription / ipo.total_subscription if ipo.total_subscription > 0 else 0

    row = {
        "Log_Issue_Size": log_issue_size,
        "QIB": ipo.qib_subscription,
        "HNI": ipo.hni_subscription,
        "RII": ipo.rii_subscription,
        "Total": ipo.total_subscription,
        "RII_to_QIB_Ratio": rii_to_qib,
        "Is_Oversubscribed": is_oversubscribed,
        "QIB_Dominance": qib_dominance,
        "Offer Price": ipo.offer_price,
        "Year": ipo.year,
    }
    return pd.DataFrame([row], columns=feature_cols)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "regressor": metadata["reg_name"],
        "regressor_r2": round(metadata["reg_r2"], 3),
        "classifier": metadata["clf_name"],
        "classifier_auc": round(metadata["clf_auc"], 3),
        "features_used": feature_cols,
        "note": (
            "Regressor R2 is moderate (~0.4) because listing-gain MAGNITUDE is "
            "noisy and shifts with market regime. The classifier is more reliable "
            "for DIRECTION (premium vs discount) than the regressor is for exact %."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(ipo: IPOInput):
    try:
        X = build_features(ipo)

        X_reg = scaler_reg.transform(X) if metadata.get("reg_needs_scaling") else X
        X_clf = scaler_clf.transform(X) if metadata.get("clf_needs_scaling") else X

        gain_pred = float(regressor.predict(X_reg)[0])
        premium_proba = float(classifier.predict_proba(X_clf)[0][1])
        premium_pred = bool(classifier.predict(X_clf)[0])

        confidence_note = (
            "High confidence" if premium_proba > 0.75 or premium_proba < 0.25
            else "Moderate confidence — subscription signals are mixed"
        )

        # Log to DB
        db = SessionLocal()
        try:
            log = PredictionLog(
                issue_size_crores=ipo.issue_size_crores,
                qib_subscription=ipo.qib_subscription,
                hni_subscription=ipo.hni_subscription,
                rii_subscription=ipo.rii_subscription,
                total_subscription=ipo.total_subscription,
                offer_price=ipo.offer_price,
                predicted_listing_gain_pct=round(gain_pred, 2),
                listed_at_premium=premium_pred,
                premium_probability=round(premium_proba, 3),
                created_at=datetime.utcnow(),
            )
            db.add(log)
            db.commit()
        finally:
            db.close()

        return PredictionResponse(
            predicted_listing_gain_pct=round(gain_pred, 2),
            listed_at_premium=premium_pred,
            premium_probability=round(premium_proba, 3),
            confidence_note=confidence_note,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predictions")
def get_predictions(limit: int = 20):
    db = SessionLocal()
    try:
        rows = (
            db.query(PredictionLog)
            .order_by(PredictionLog.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "issue_size_crores": r.issue_size_crores,
                "total_subscription": r.total_subscription,
                "predicted_listing_gain_pct": r.predicted_listing_gain_pct,
                "listed_at_premium": r.listed_at_premium,
                "premium_probability": r.premium_probability,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    finally:
        db.close()
