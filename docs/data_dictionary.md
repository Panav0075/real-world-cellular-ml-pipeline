# Data Dictionary

| Column | Description |
|---|---|
| `timestamp` | Time at which the synthetic measurement was recorded |
| `cell_id` | Anonymous cellular site/cell identifier |
| `network_type` | LTE or 5G |
| `frequency_band` | Cellular frequency-band label |
| `rsrp_dbm` | Reference Signal Received Power |
| `rsrq_db` | Reference Signal Received Quality |
| `sinr_db` | Signal-to-Interference-plus-Noise Ratio |
| `rssi_dbm` | Received Signal Strength Indicator |
| `latency_ms` | Simulated round-trip latency in milliseconds |
| `packet_loss_pct` | Simulated percentage of packets lost |
| `handover_count` | Number of recent cell handovers |
| `throughput_mbps` | Synthetic downstream throughput and prediction target |

## Important note

The measurements are synthetic. Their ranges and relationships are intended for
demonstrating an ML workflow, not for benchmarking or characterizing a real
mobile network.
