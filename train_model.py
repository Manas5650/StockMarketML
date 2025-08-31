# train_model.py
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

REQUIRED_FEATURES = ["open", "high", "low", "volume"]

def train(csv_path, out_path):
    print(f"Training on {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Date cleanup
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Make sure required columns are numeric
    for col in REQUIRED_FEATURES + ["close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_FEATURES + ["close"]).copy()

    X = df[REQUIRED_FEATURES].to_numpy(dtype=float)
    y = df["close"].to_numpy(dtype=float)

    # Train simple random forest
    model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")

    # Save model
    joblib.dump(model, out_path)
    print(f"✅ Model saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--out", required=True, help="Output model path")
    args = parser.parse_args()

    train(args.csv, args.out)