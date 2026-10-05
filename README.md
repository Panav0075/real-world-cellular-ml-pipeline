# Real-World Cellular Network ML Pipeline

A practical machine learning project built around a problem that is usually skipped in tutorials: **real data is messy**.

Most ML examples start with a clean CSV where every column already has the right type, missing values are handled, and the target is ready for training. This project starts earlier in the process. It works with raw-style cellular network measurements that contain missing values, duplicates, inconsistent labels, malformed records, and unusual signal readings.

The goal is to turn that data into something reliable enough to train and evaluate a machine learning model.

> **Portfolio note:** The sample data included in this repository is synthetic raw-style cellular telemetry. It is designed to reproduce common data-quality problems without publishing proprietary or customer data.

## What this project does

The pipeline:

1. Loads raw cellular network measurements.
2. Checks the dataset for common quality problems.
3. Cleans missing, duplicated, inconsistent, and invalid records.
4. Creates useful features from timestamps and signal measurements.
5. Trains several regression models.
6. Compares them against a simple baseline.
7. Saves the best model and evaluation results.
8. Includes automated tests for the most important preprocessing logic.

The prediction task used here is **downlink throughput estimation**.

## Why cellular data?

Network measurements are a good example of real-world ML data because values come from different devices, locations, cells, network technologies, and collection systems.

A raw record may contain measurements such as:

- RSRP — reference signal received power
- RSRQ — reference signal received quality
- SINR — signal-to-interference-plus-noise ratio
- RSSI — received signal strength
- latency
- packet loss
- network type
- frequency band
- cell identifier
- handover count
- timestamp

These values can help explain network performance, but they need to be checked carefully before they are used by a model.

## Architecture

```text
                     Raw cellular measurements
                               |
                               v
                    +---------------------+
                    |     Data Loader     |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |  Quality Validation |
                    | missing / duplicate |
                    | ranges / data types |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |    Preprocessing    |
                    | clean + normalize   |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    | Feature Engineering |
                    | time + signal data  |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |   Train/Test Split  |
                    +----------+----------+
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
       Linear Model      Random Forest    Gradient Boosting
             |                 |                 |
             +-----------------+-----------------+
                               |
                               v
                    +---------------------+
                    | Model Evaluation    |
                    | MAE / RMSE / R²     |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    | Best Model + Report |
                    +---------------------+
```

## Repository structure

```text
real-world-cellular-ml/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── LICENSE
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── sample_cell_measurements.csv
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── ingest.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   └── test_features.py
│
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── modeling.md
│
├── reports/
│   └── model_evaluation.md
│
└── models/
```

## Example of messy input

A few records may look perfectly normal:

```text
timestamp,cell_id,network_type,rsrp_dbm,rsrq_db,sinr_db,latency_ms
2026-01-04 08:10:00,CELL_104,LTE,-91,-10,14.2,35
```

Others may contain problems:

```text
2026-01-04 08:15:00,,lte,-140,-10,unknown,42
bad_timestamp,CELL_209,4G,-88,,18.1,31
2026-01-04 08:25:00,CELL_104,LTE,-250,-9,16.0,39
```

The preprocessing pipeline handles these cases instead of assuming that the input is already clean.

## Data cleaning

The project checks for:

- duplicate rows
- missing values
- malformed timestamps
- numeric values stored as text
- invalid signal measurements
- inconsistent network labels such as `LTE`, `lte`, and `4G`
- impossible packet-loss percentages
- unrealistic latency and throughput values

Invalid measurements are converted to missing values and then handled by the preprocessing pipeline rather than silently being treated as valid observations.

## Feature engineering

The pipeline creates additional features including:

- hour of day
- day of week
- weekend indicator
- signal quality score
- latency-to-signal relationship
- packet-loss interaction
- network technology
- frequency band

The intention is not to create hundreds of features. It is to create a small set whose meaning can be explained.

## Models

Four approaches are evaluated:

| Model | Why it is included |
|---|---|
| Mean Baseline | Shows whether ML actually improves over a simple prediction |
| Linear Regression | Easy-to-understand statistical benchmark |
| Random Forest | Handles nonlinear relationships and interactions |
| Gradient Boosting | Strong tree-based model for tabular data |

The repository does not hard-code impressive-looking scores. Run the training pipeline and the evaluation table is generated from the actual model outputs.

## Running the project

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate a fresh raw-style dataset (optional)

A sample dataset is already included.

```bash
python -m src.generate_data
```

### 4. Train and evaluate

```bash
python -m src.train
```

This creates:

```text
data/processed/clean_cell_measurements.csv
models/best_model.joblib
reports/metrics.json
reports/model_evaluation.md
```

### 5. Run tests

```bash
pytest -q
```

### 6. Make predictions

```bash
python -m src.predict data/raw/sample_cell_measurements.csv
```

## Results

The repository was run end-to-end against the included sample dataset. These are the generated test-set results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 35.16 | 42.25 | -0.003 |
| Linear Regression | 11.13 | 14.26 | 0.886 |
| Random Forest | 12.44 | 16.08 | 0.855 |
| Gradient Boosting | 11.79 | 15.19 | 0.870 |

**Selected model:** Linear Regression (lowest test RMSE).

The baseline is included deliberately: it makes it clear whether the trained models are learning useful relationships rather than simply producing plausible-looking predictions.

## Evaluation

The training script calculates:

- **MAE** — average absolute prediction error
- **RMSE** — gives more weight to large errors
- **R²** — how much variation in throughput is explained by the model

The generated report makes it easy to compare the models without manually copying metrics into the README.

## Engineering decisions

### Why keep the raw data?

The raw file makes the cleaning work visible. A recruiter or reviewer can inspect the input and see why preprocessing is necessary.

### Why use a baseline?

A complicated model is not automatically useful. The mean baseline provides a minimum reference point that trained models should beat.

### Why use a scikit-learn Pipeline?

The same preprocessing steps used during training are saved with the model. This reduces the chance of applying different transformations when new data is used for prediction.

### Why use synthetic data?

Real telecom datasets can contain proprietary network information and location data. Synthetic raw-style data lets the complete pipeline remain public and reproducible.

## Limitations

This repository is a portfolio implementation, not a live telecom optimization platform.

The included dataset is synthetic and does not represent the full complexity of measurements collected from a commercial cellular network. A production version would also need stronger schema contracts, data-drift monitoring, model monitoring, access controls, experiment tracking, and integration with a real data platform.

## Possible next steps

- add MLflow experiment tracking
- add drift detection
- support larger Parquet datasets
- add geographic/network-cell analysis
- expose predictions through FastAPI
- containerize the training and inference workflow
- add CI checks with GitHub Actions
- compare time-based validation against random splitting

## Tech stack

**Python · Pandas · NumPy · scikit-learn · Joblib · PyYAML · Pytest**

