# test.py
from evtp.generator import TelemetryGenerator
from evtp.etl import ETLPipeline
from evtp.model import FailureModel
import os

# ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1) generate data
TelemetryGenerator(["EV001","EV002"], hz=5).to_csv("data/raw.csv", rows=2000)
print("✅ generated data → data/raw.csv")

# 2) ETL (CSV -> SQLite with engineered features)
etl = ETLPipeline(db_path="data/ev_telemetry.db")
raw_n, feat_n = etl.run("data/raw.csv")
print(f"✅ etl complete: raw={raw_n}, features={feat_n}")

# 3) train model
fm = FailureModel(db_path="data/ev_telemetry.db", outdir="models")
metrics = fm.train()
print(f"✅ model trained: AP={metrics['average_precision']:.3f}  AUC={metrics['auc']:.3f}")
print("Artifacts:", os.listdir("models"))