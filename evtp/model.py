from __future__ import annotations
import os, json, sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

class FailureModel:
    """
        Loads engineered features from SQLite, creates synthetic fialure labels, trains a model, and saves artifacts for later inference

    """

    def __init__(self, db_path: str = "data/ev_telemetry.db", outdir: str = 'models'):
        self.db_path = db_path
        self.outdir = outdir
        os.makedirs(outdir, exist_ok = True)


    def _load_features(self) -> pd.DataFrame:
        con = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM features", con, parse_dates=["timestamp"])
        con.close()
        if df.empty:
            raise RuntimeError("No rows found in features table. Run ETL first.")
        return df
    
    # ---------- synthetic labels (since data is simulated) ----------
    @staticmethod
    def synth_labels(df: pd.DataFrame) -> pd.Series:
        z = (
            0.015*df["thermal_stress"].clip(lower=0) +
            0.0008*df["power_stress"].clip(lower=0) +
            (-0.02*df["soc_pct"]) +
            (0.05*(100 - df["tire_wear_pct"])) +
            (0.07*(100 - df["brake_wear_pct"]))
        )
        p = 1/(1+np.exp(-0.02*(z - np.median(z))))
        rng = np.random.default_rng(42)
        return (rng.random(len(df)) < p).astype(int)
    
    # ---------- training ----------
    def train(self) -> dict:
        df = self._load_features()
        exclude = {"timestamp", "vin"}
        Xcols = [c for c in df.columns if c not in exclude]
        X = df[Xcols].fillna(0.0)
        y = self.synth_labels(df)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42
        )
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)[:, 1]
        ap = average_precision_score(yte, proba)
        auc = roc_auc_score(yte, proba)

        # Save artifacts
        joblib.dump(clf, os.path.join(self.outdir, "model.joblib"))
        with open(os.path.join(self.outdir, "feature_cols.json"), "w") as f:
            json.dump(Xcols, f, indent=2)

        return {"average_precision": float(ap), "auc": float(auc)}

    # ---------- inference helpers ----------
    def load(self):
        model = joblib.load(os.path.join(self.outdir, "model.joblib"))
        feats = json.load(open(os.path.join(self.outdir, "feature_cols.json")))
        return model, feats