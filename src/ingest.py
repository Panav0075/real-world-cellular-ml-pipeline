from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "cell_id",
    "network_type",
    "frequency_band",
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "rssi_dbm",
    "latency_ms",
    "packet_loss_pct",
    "handover_count",
    "throughput_mbps",
}


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset was not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df
