# evtp/service.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class PredictionService:
    """
    Loads the trained model + feature schema and scores the latest
    feature rows for a given VIN from SQLite.
    """
    db_path: str = "data/ev_telemetry.db"
    model_path: str = "models/model.joblib"
    feature_cols_path: str = "models/feature_cols.json"

    def __post_init__(self):
        self.model = joblib.load(self.model_path)
        with open(self.feature_cols_path, "r") as f:
            self.feature_cols: List[str] = json.load(f)

    def fetch_latest_features(self, vin: str, n: int = 1) -> pd.DataFrame:
        con = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql(
                "SELECT * FROM features WHERE vin = ? ORDER BY timestamp DESC LIMIT ?",
                con,
                params=[vin, n],
                parse_dates=["timestamp"],
            )
        finally:
            con.close()

        if df.empty:
            raise ValueError(f"No data found for VIN={vin}. Did you run generator+ETL?")

        # Align columns exactly as model expects (same order)
        X = df.reindex(columns=self.feature_cols).fillna(0.0)
        return X

    def predict_risk(self, vin: str, n: int = 1) -> List[float]:
        X = self.fetch_latest_features(vin, n)
        proba = self.model.predict_proba(X)[:, 1]
        return proba.tolist()


# ----- FastAPI wiring -----

app = FastAPI(title="EV Telemetry Predict API")
svc = PredictionService()


class PredictRequest(BaseModel):
    vin: str
    n: int = 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    risk = svc.predict_risk(req.vin, req.n)
    return {"vin": req.vin, "n": len(risk), "risk": risk}