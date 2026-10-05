import pandas as pd
from src.preprocessing import clean_data


def sample_frame():
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01 10:00:00", "2026-01-01 10:05:00"],
            "cell_id": ["CELL_1", "CELL_2"],
            "network_type": ["4G", "5g"],
            "frequency_band": ["B2", "n77"],
            "rsrp_dbm": [-90, -250],
            "rsrq_db": [-10, -12],
            "sinr_db": [15, "unknown"],
            "rssi_dbm": [-72, -80],
            "latency_ms": [30, 40],
            "packet_loss_pct": [1, 2],
            "handover_count": [0, 1],
            "throughput_mbps": [80, 100],
        }
    )


def test_network_labels_are_normalized():
    cleaned = clean_data(sample_frame())
    assert cleaned["network_type"].tolist() == ["LTE", "5G"]


def test_invalid_signal_is_replaced_with_missing():
    cleaned = clean_data(sample_frame())
    assert pd.isna(cleaned.loc[1, "rsrp_dbm"])
    assert pd.isna(cleaned.loc[1, "sinr_db"])
