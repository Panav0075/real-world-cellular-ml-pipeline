import pandas as pd
from src.features import add_features


def test_time_features_are_created():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-03 14:00:00"]),
            "rsrp_dbm": [-90.0],
            "rsrq_db": [-10.0],
            "sinr_db": [15.0],
            "latency_ms": [25.0],
            "packet_loss_pct": [1.0],
        }
    )
    result = add_features(df)
    assert result.loc[0, "hour"] == 14
    assert result.loc[0, "is_weekend"] == 1
    assert "signal_quality_score" in result.columns
