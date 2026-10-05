import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "rssi_dbm",
    "latency_ms",
    "packet_loss_pct",
    "handover_count",
    "throughput_mbps",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy().drop_duplicates()

    cleaned["network_type"] = (
        cleaned["network_type"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"4G": "LTE"})
    )

    cleaned["cell_id"] = cleaned["cell_id"].astype("string").str.strip()
    cleaned["frequency_band"] = cleaned["frequency_band"].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")

    # Convert physically implausible measurements to missing values.
    cleaned.loc[~cleaned["rsrp_dbm"].between(-160, -40), "rsrp_dbm"] = np.nan
    cleaned.loc[~cleaned["rsrq_db"].between(-30, 0), "rsrq_db"] = np.nan
    cleaned.loc[~cleaned["sinr_db"].between(-30, 50), "sinr_db"] = np.nan
    cleaned.loc[~cleaned["latency_ms"].between(0, 5000), "latency_ms"] = np.nan
    cleaned.loc[~cleaned["packet_loss_pct"].between(0, 100), "packet_loss_pct"] = np.nan

    # A supervised training row must have a valid timestamp and target.
    cleaned = cleaned.dropna(subset=["timestamp", "throughput_mbps"])
    cleaned = cleaned[cleaned["throughput_mbps"] > 0]

    # scikit-learn categorical imputers expect np.nan rather than pandas pd.NA.
    for column in ["cell_id", "network_type", "frequency_band"]:
        cleaned[column] = cleaned[column].astype(object).where(cleaned[column].notna(), np.nan)

    return cleaned.reset_index(drop=True)
