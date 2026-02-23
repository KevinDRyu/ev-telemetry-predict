# evtp/service.py
from __future__ import annotations
from datetime import datetime
from typing import Optional
from evtp.etl import ETLPipeline
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
    db_path: str = "data/ev_telemetry.db"
    model_path: str = "models/model.joblib"
    feature_cols_path: str = "models/feature_cols.json"

    def __post_init__(self):
        self.model = joblib.load(self.model_path)
        with open(self.feature_cols_path, "r") as f:
            self.feature_cols: List[str] = json.load(f)

        # reuse your ETL feature logic
        self.etl = ETLPipeline(db_path=self.db_path)

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables_exist(self) -> None:
        """
        Make sure raw/features exist. If you already ran test.py, they will.
        This is mostly safety so /ingest doesn't crash on a fresh DB.
        """
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS raw (
                    timestamp TEXT,
                    vin TEXT,
                    speed_kmh REAL,
                    soc_pct REAL,
                    battery_temp_c REAL,
                    motor_current_a REAL,
                    inverter_temp_c REAL,
                    ambient_temp_c REAL,
                    tire_wear_pct REAL,
                    brake_wear_pct REAL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    timestamp TEXT,
                    vin TEXT
                );
            """)
            # indices
            cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_vin_ts ON raw(vin, timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_vin_ts ON features(vin, timestamp);")
            con.commit()
        finally:
            con.close()

    def ingest_records(self, records: list[TelemetryRecord], recompute_features: bool = True) -> dict:
        self._ensure_tables_exist()

        # Convert to DataFrame (match raw schema)
        df = pd.DataFrame([r.model_dump() for r in records])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["vin", "timestamp"])

        con = self._connect()
        try:
            # append to raw
            df.to_sql("raw", con, if_exists="append", index=False)
        finally:
            con.close()

        vins = sorted(df["vin"].unique().tolist())

        if recompute_features:
            self._recompute_features_for_vins(vins)

        return {"inserted_rows": int(len(df)), "vins": vins, "recomputed_features": bool(recompute_features)}

    def _recompute_features_for_vins(self, vins: list[str]) -> None:
        """
        Simple & reliable: re-read ALL raw rows for each VIN, re-run feature engineering,
        then replace features for those VINs.
        """
        con = self._connect()
        try:
            # pull raw for these vins
            placeholders = ",".join(["?"] * len(vins))
            raw = pd.read_sql(
                f"SELECT * FROM raw WHERE vin IN ({placeholders}) ORDER BY vin, timestamp",
                con,
                params=vins,
                parse_dates=["timestamp"],
            )

            if raw.empty:
                return

            feat = self.etl.feature_engineer(raw)

            # delete old features for these vins
            cur = con.cursor()
            cur.execute(f"DELETE FROM features WHERE vin IN ({placeholders})", vins)
            con.commit()

            # write new features (append)
            feat.to_sql("features", con, if_exists="append", index=False)

            # indices (safe to call repeatedly)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_vin_ts ON features(vin, timestamp);")
            con.commit()
        finally:
            con.close()

    def fetch_latest_features(self, vin: str, n: int = 1) -> pd.DataFrame:
        con = self._connect()
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
            raise ValueError(f"No data found for VIN={vin}. Did you ingest or run ETL?")

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
class TelemetryRecord(BaseModel):
    timestamp: datetime
    vin: str
    speed_kmh: float
    soc_pct: float
    battery_temp_c: float
    motor_current_a: float
    inverter_temp_c: float
    ambient_temp_c: float
    tire_wear_pct: float
    brake_wear_pct: float

class IngestRequest(BaseModel):
    records: list[TelemetryRecord]
    recompute_features: bool = True

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    risk = svc.predict_risk(req.vin, req.n)
    return {"vin": req.vin, "n": len(risk), "risk": risk}

@app.post("/ingest")
def ingest(req: IngestRequest):
    result = svc.ingest_records(req.records, recompute_features=req.recompute_features)
    return {"status": "ok", **result}