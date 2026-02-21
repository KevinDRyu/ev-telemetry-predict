from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List
import csv, os, random

def _clamp(x, lo ,hi):
    return max(lo, min(hi,x))

@dataclass
class VehicleState:
    soc: float = 100.0
    speed: float = 0.0         # km/h
    batt: float = 25.0         # °C
    motor: float = 10.0        # arbitrary load component
    inverter: float = 20.0     # °C
    ambient: float = 20.0      # °C
    tire: float = 100.0        # % remaining
    brake: float = 100.0       # % remaining

@dataclass
class TelemetryGenerator:
    vins: List[str]
    hz: float = 2.0
    seed: int = 42
    states: Dict[str, VehicleState] = field(init = False)

    def __post_init__(self):
        random.seed(self.seed)
        self.states = {vin: VehicleState() for vin in self.vins}
        
    def _step(self, vin: str) -> Dict[str, float]:
        s = self.states[vin]
        # speed random walk
        s.speed = _clamp(s.speed + random.uniform(-2, 3), 0, 160)
        # soc drains with speed + noise
        s.soc = _clamp(s.soc - (0.0007 * s.speed) + random.uniform(-0.02, 0.02), 0, 100)
       # temps (stable + realistic)
        target_batt = s.ambient + 10 + 0.05 * s.speed
        target_inv  = s.ambient + 15 + 0.07 * s.speed

        # move toward target (cooling/heating dynamics)
        s.batt = _clamp(s.batt + 0.08 * (target_batt - s.batt) + random.uniform(-0.2, 0.2), -5, 80)
        s.inverter = _clamp(s.inverter + 0.10 * (target_inv - s.inverter) + random.uniform(-0.3, 0.3), -5, 120)

        # optional: motor current should not accumulate either
        target_motor = 20 + 1.2 * s.speed
        s.motor = _clamp(s.motor + 0.12 * (target_motor - s.motor) + random.uniform(-3, 3), 0, 400)
        # wear
        s.tire = _clamp(s.tire - random.uniform(0.00002, 0.00008), 0, 100)
        s.brake = _clamp(s.brake - random.uniform(0.00005, 0.00012), 0, 100)
        # ambient drift
        s.ambient = _clamp(s.ambient + random.uniform(-0.05, 0.05), -10, 45)

        return {
            "speed_kmh": round(s.speed, 2),
            "soc_pct": round(s.soc, 2),
            "battery_temp_c": round(s.batt, 2),
            "motor_current_a": round(100 + s.motor, 2),  # proxy for load
            "inverter_temp_c": round(s.inverter, 2),
            "ambient_temp_c": round(s.ambient, 2),
            "tire_wear_pct": round(s.tire, 2),
            "brake_wear_pct": round(s.brake, 2),
        }
    
    def stream_rows(self, rows: int, start: datetime | None = None) -> Iterable[Dict]:
        t = start or datetime.utcnow()
        step = timedelta(seconds=1.0 / self.hz)
        for _ in range(rows):
            for vin in self.vins:
                vals = self._step(vin)
                yield {"timestamp": t.isoformat(), "vin": vin, **vals}
            t += step

    def to_csv(self, path: str, rows: int, start: datetime | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = ["timestamp","vin","speed_kmh","soc_pct","battery_temp_c",
                  "motor_current_a","inverter_temp_c","ambient_temp_c",
                  "tire_wear_pct","brake_wear_pct"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader()
            for r in self.stream_rows(rows, start=start):
                w.writerow(r)

