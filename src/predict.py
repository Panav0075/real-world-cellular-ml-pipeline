import argparse
from pathlib import Path

import joblib

from .features import add_features
from .ingest import load_data
from .preprocessing import clean_data


def predict(input_path: str, model_path="models/best_model.joblib"):
    artifact = joblib.load(model_path)
    df = add_features(clean_data(load_data(input_path)))

    predictions = artifact["pipeline"].predict(df[artifact["features"]])
    output = df.copy()
    output["predicted_throughput_mbps"] = predictions
    return output


def main():
    parser = argparse.ArgumentParser(description="Predict cellular throughput.")
    parser.add_argument("input", help="Path to a CSV containing cellular measurements")
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()

    result = predict(args.input, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Predictions saved -> {output}")


if __name__ == "__main__":
    main()
