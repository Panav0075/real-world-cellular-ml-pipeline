from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT = Path("data/raw/sample_cell_measurements.csv")
RANDOM_STATE = 42


def generate_dataset(n_rows: int = 1200, output_path: Path = OUTPUT) -> pd.DataFrame:
    """Generate synthetic cellular telemetry and deliberately add data-quality issues."""
    rng = np.random.default_rng(RANDOM_STATE)

    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
    network = rng.choice(["LTE", "5G"], size=n_rows, p=[0.58, 0.42])
    rsrp = rng.normal(-94, 13, n_rows)
    rsrq = rng.normal(-11, 3, n_rows)
    sinr = rng.normal(13, 7, n_rows)
    rssi = rsrp + rng.normal(18, 5, n_rows)
    latency = np.maximum(5, rng.normal(42, 15, n_rows) - (network == "5G") * 9)
    packet_loss = np.clip(rng.gamma(1.4, 0.7, n_rows), 0, 12)
    handovers = rng.poisson(0.7, n_rows)
    band = np.where(
        network == "5G",
        rng.choice(["n41", "n77"], n_rows),
        rng.choice(["B2", "B4", "B12", "B66"], n_rows),
    )
    cell_ids = rng.choice([f"CELL_{i:03d}" for i in range(100, 130)], n_rows)

    # A synthetic relationship with noise. This is not intended to model a real operator.
    throughput = (
        55
        + (network == "5G") * 55
        + (rsrp + 110) * 1.35
        + sinr * 2.1
        - latency * 0.42
        - packet_loss * 5.2
        - handovers * 1.8
        + rng.normal(0, 12, n_rows)
    )
    throughput = np.clip(throughput, 1, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "cell_id": cell_ids,
            "network_type": network,
            "frequency_band": band,
            "rsrp_dbm": np.round(rsrp, 2),
            "rsrq_db": np.round(rsrq, 2),
            "sinr_db": np.round(sinr, 2),
            "rssi_dbm": np.round(rssi, 2),
            "latency_ms": np.round(latency, 2),
            "packet_loss_pct": np.round(packet_loss, 3),
            "handover_count": handovers,
            "throughput_mbps": np.round(throughput, 2),
        }
    )

    # Deliberately introduce raw-data problems.
    missing_idx = rng.choice(df.index, 55, replace=False)
    df.loc[missing_idx[:20], "rsrp_dbm"] = np.nan
    df.loc[missing_idx[20:35], "cell_id"] = np.nan
    df.loc[missing_idx[35:], "rsrq_db"] = np.nan

    label_idx = rng.choice(df.index, 35, replace=False)
    df.loc[label_idx[:12], "network_type"] = "lte"
    df.loc[label_idx[12:24], "network_type"] = "4G"
    df.loc[label_idx[24:], "network_type"] = "5g"

    malformed_idx = rng.choice(df.index, 8, replace=False)
    df.loc[malformed_idx, "timestamp"] = "bad_timestamp"

    bad_signal_idx = rng.choice(df.index, 8, replace=False)
    df.loc[bad_signal_idx, "rsrp_dbm"] = rng.choice([-250, 20, 999], len(bad_signal_idx))

    bad_loss_idx = rng.choice(df.index, 5, replace=False)
    df.loc[bad_loss_idx, "packet_loss_pct"] = rng.choice([-5, 120, 999], len(bad_loss_idx))

    text_numeric_idx = rng.choice(df.index, 7, replace=False)
    df["sinr_db"] = df["sinr_db"].astype(object)
    df.loc[text_numeric_idx, "sinr_db"] = "unknown"

    # Add duplicates.
    duplicates = df.sample(15, random_state=RANDOM_STATE)
    df = pd.concat([df, duplicates], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    frame = generate_dataset()
    print(f"Generated {len(frame):,} raw records -> {OUTPUT}")
