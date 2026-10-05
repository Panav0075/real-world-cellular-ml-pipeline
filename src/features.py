import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    # Simple interpretable combinations of network measurements.
    result["signal_quality_score"] = (
        (result["rsrp_dbm"] + 140) * 0.45
        + (result["rsrq_db"] + 30) * 0.25
        + (result["sinr_db"] + 30) * 0.30
    )
    result["latency_signal_ratio"] = result["latency_ms"] / (
        result["sinr_db"].abs() + 1.0
    )
    result["loss_latency_interaction"] = (
        result["packet_loss_pct"] * result["latency_ms"]
    )

    return result.replace([np.inf, -np.inf], np.nan)
