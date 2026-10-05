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


def quality_summary(df: pd.DataFrame) -> dict:
    numeric_invalid = {}
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")
        numeric_invalid[column] = int((converted.isna() & df[column].notna()).sum())

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")

    return {
        "rows": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
        "malformed_timestamps": int(timestamps.isna().sum()),
        "non_numeric_values": {k: v for k, v in numeric_invalid.items() if v},
    }


def valid_ranges(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for rows with plausible values after numeric conversion."""
    rsrp = pd.to_numeric(df["rsrp_dbm"], errors="coerce")
    rsrq = pd.to_numeric(df["rsrq_db"], errors="coerce")
    sinr = pd.to_numeric(df["sinr_db"], errors="coerce")
    latency = pd.to_numeric(df["latency_ms"], errors="coerce")
    loss = pd.to_numeric(df["packet_loss_pct"], errors="coerce")
    throughput = pd.to_numeric(df["throughput_mbps"], errors="coerce")

    return (
        (rsrp.between(-160, -40) | rsrp.isna())
        & (rsrq.between(-30, 0) | rsrq.isna())
        & (sinr.between(-30, 50) | sinr.isna())
        & (latency.between(0, 5000) | latency.isna())
        & (loss.between(0, 100) | loss.isna())
        & (throughput.gt(0) | throughput.isna())
    )
