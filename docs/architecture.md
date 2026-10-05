# Architecture

## Pipeline overview

The project is split into small stages instead of placing all logic inside one
notebook.

### Ingestion

`src/ingest.py` loads CSV data and verifies that the columns required by the
pipeline exist.

### Validation

`src/validation.py` summarizes missing values, duplicates, malformed timestamps,
and numeric conversion problems.

### Cleaning

`src/preprocessing.py` normalizes network labels, converts measurements to their
expected types, removes duplicate records, and converts implausible measurements
to missing values.

### Feature engineering

`src/features.py` adds time-based and signal-related features.

### Training

`src/train.py` builds scikit-learn pipelines so preprocessing and model inference
remain together. Four approaches are evaluated using the same train/test split.

### Model selection

The candidate with the lowest test RMSE is saved to `models/best_model.joblib`.

### Inference

`src/predict.py` applies the same cleaning, feature engineering, and saved
scikit-learn pipeline to new CSV data.

## Design principle

The main design goal is reproducibility. Cleaning logic lives in Python modules,
model preprocessing is stored inside the fitted pipeline, and the random state is
fixed in configuration.
