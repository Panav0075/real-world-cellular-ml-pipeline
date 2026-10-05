# Modeling Approach

## Target

The supervised-learning target is `throughput_mbps`.

## Baseline

A `DummyRegressor` predicting the training-set mean is included. This provides a
simple reference before comparing actual ML models.

## Candidate models

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Preprocessing

Numeric features use median imputation followed by standardization. Categorical
features use most-frequent imputation and one-hot encoding.

All transformations are part of a scikit-learn `Pipeline`, which means the same
transformations are available when the saved model is used later.

## Metrics

- MAE
- RMSE
- R²

The best candidate is selected using the lowest test RMSE.

## Validation limitation

The current implementation uses a reproducible random train/test split. A more
realistic production experiment could use time-based validation or hold out
entire cells/regions to measure generalization under harder conditions.
