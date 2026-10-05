import pandas as pd
from src.validation import quality_summary


def test_quality_summary_detects_duplicate_and_bad_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": ["bad", "bad"],
            "rsrp_dbm": [-90, -90],
            "rsrq_db": [-10, -10],
            "sinr_db": [10, 10],
            "rssi_dbm": [-70, -70],
            "latency_ms": [30, 30],
            "packet_loss_pct": [1, 1],
            "handover_count": [0, 0],
            "throughput_mbps": [50, 50],
        }
    )
    summary = quality_summary(df)
    assert summary["duplicate_rows"] == 1
    assert summary["malformed_timestamps"] == 2
