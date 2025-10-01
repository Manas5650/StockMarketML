# app.py — dataset-aware backend with per-dataset models
from flask import Flask, request, jsonify, render_template, make_response
import io, base64, logging, os
import joblib
import numpy as np
import pandas as pd

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functools import lru_cache
from flask_cors import CORS  # pip install flask-cors
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------- app setup ----------------
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- helpers ----------------
def json_nocache(payload, status=200):
    resp = make_response(jsonify(payload), status)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

@app.after_request
def add_no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

def safe_error(msg="Server error", code=500, exc_info=True):
    if exc_info:
        logger.exception(msg)
    else:
        logger.error(msg)
    return json_nocache({"error": msg}, status=code)

def get_dataset_key_from_request() -> str:
    key = (request.args.get("dataset")
           or (request.get_json(silent=True) or {}).get("dataset")
           or "DEFAULT")
    return str(key).upper()

# Safe MAPE calculation
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# ---------------- model paths ----------------
MODEL_PATHS = {
    "DEFAULT": "random_forest_model_small.pkl",
    "NIFTY":   "models/nifty_rf.pkl",
    "SP500":   "models/sp500_rf.pkl",
    "TESLA":   "models/tesla_rf.pkl"
}
MODELS = {}

def get_model(dataset_key: str):
    key = (dataset_key or "DEFAULT").upper()
    path = MODEL_PATHS.get(key, MODEL_PATHS["DEFAULT"])
    if key not in MODELS:
        if not os.path.exists(path):
            raise RuntimeError(f"Model file not found: {path}")
        MODELS[key] = joblib.load(path)
        logger.info("Loaded model for %s from %s", key, path)
    return MODELS[key]

# ---------------- dataset paths ----------------
DATASETS = {
    "DEFAULT": "DEFAULT.csv",
    "NIFTY":   "datasets/NIFTY.csv",
    "SP500":   "datasets/SP500.csv",
    "TESLA":   "datasets/TESLA.csv",
}
REQUIRED_FEATURES = ["open", "high", "low", "volume"]

def _load_and_clean_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise RuntimeError(f"{path} must have a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "volume", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.mean(numeric_only=True))
    return df

@lru_cache(maxsize=8)
def _cached_df_for_key(dataset_key: str) -> pd.DataFrame:
    key = (dataset_key or "DEFAULT").upper()
    if key not in DATASETS:
        key = "DEFAULT"
    return _load_and_clean_df(DATASETS[key])

def get_df(dataset_key: str) -> pd.DataFrame:
    return _cached_df_for_key(dataset_key).copy()

# ---------------- routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    try:
        dataset = get_dataset_key_from_request()
        d = get_df(dataset)
        symbols = (d["Name"].astype(str).str.upper().nunique()
                   if "Name" in d.columns else 1)
        model_path = MODEL_PATHS.get(dataset, MODEL_PATHS["DEFAULT"])
        return json_nocache({
            "ok": True,
            "dataset": dataset,
            "rows": int(len(d)),
            "symbols": int(symbols),
            "min_date": str(d["date"].min().date()),
            "max_date": str(d["date"].max().date()),
            "model_loaded": os.path.exists(model_path),
            "model_path": model_path
        })
    except Exception:
        return safe_error("Health check failed")

@app.route("/datasets")
def datasets():
    try:
        return json_nocache(sorted(DATASETS.keys()))
    except Exception:
        return safe_error("Could not list datasets")

@app.route("/symbols", methods=["GET", "POST"])
def symbols():
    try:
        dataset = get_dataset_key_from_request()
        d = get_df(dataset)
        if "Name" in d.columns:
            syms = sorted(d["Name"].dropna().astype(str).str.upper().unique().tolist())
        else:
            syms = [dataset]
        return json_nocache(syms)
    except Exception:
        return safe_error("Could not fetch symbols")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
        dataset = get_dataset_key_from_request()
        model = get_model(dataset)

        for k in REQUIRED_FEATURES:
            if k not in data:
                return json_nocache({"error": f"Missing key: {k}"}, 400)

        feats = np.array([[float(data["open"]), float(data["high"]),
                           float(data["low"]),  float(data["volume"])]], dtype=float)
        pred = float(model.predict(feats)[0])
        return json_nocache({"prediction": pred})
    except Exception:
        return safe_error("Prediction failed")

# -------- predict_range -------------
@app.route("/predict_range", methods=["POST"])
def predict_range():
    try:
        p = request.get_json(force=True) or {}
        dataset = (p.get("dataset") or "DEFAULT").upper()
        model = get_model(dataset)

        # Validate dates
        start_date = pd.to_datetime(p.get("start_date"), errors="coerce")
        end_date   = pd.to_datetime(p.get("end_date"), errors="coerce")
        if pd.isna(start_date) or pd.isna(end_date):
            return json_nocache({"error": "Invalid or missing start_date/end_date (use YYYY-MM-DD)"}, 400)
        if start_date > end_date:
            return json_nocache({"error": "start_date must be <= end_date"}, 400)

        symbol = (p.get("symbol") or "").strip().upper()
        d = get_df(dataset)
        MIN_DATE = d["date"].min().date()
        MAX_DATE = d["date"].max().date()

        if "Name" in d.columns:
            # d["Name"] = d["Name"].astype(str).str.upper().str.strip()
            if symbol:
                d = d[d["Name"].astype(str).str.upper() == symbol].copy()
                if d.empty:
                    return json_nocache({"error": f"No data for symbol '{symbol}'"}, 404)
            else:
                return json_nocache({"error": f"Please select a symbol for dataset {dataset}"}, 400)
        else:
            symbol = dataset
            

        d = d[(d["date"] >= start_date) & (d["date"] <= end_date)].copy()
        if d.empty:
            return json_nocache({"error": f"No data in this range. Available: {MIN_DATE} to {MAX_DATE}"}, 404)

        X = d[REQUIRED_FEATURES].to_numpy(dtype=float)
        y_true = d["close"].to_numpy(dtype=float)
        y_pred = model.predict(X)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        mape = safe_mape(y_true, y_pred)

        plt.figure(figsize=(9, 5))
        label = (symbol or dataset)
        plt.plot(d["date"], y_true, label=f"{label} Actual", linewidth=1.5)
        plt.plot(d["date"], y_pred, label=f"{label} Predicted", linestyle="--", linewidth=1.5)
        plt.legend(); plt.xticks(rotation=45); plt.tight_layout()

        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150)
        buf.seek(0); img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8"); plt.close()

        return json_nocache({"graph": img_b64, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape})
    except Exception:
        return safe_error("Failed to create range prediction")

# -------- compare_range -------------
@app.route("/compare_range", methods=["POST"])
def compare_range():
    try:
        p = request.get_json(force=True) or {}
        dataset = (p.get("dataset") or "DEFAULT").upper()
        start_date = pd.to_datetime(p.get("start_date"), errors="coerce")
        end_date   = pd.to_datetime(p.get("end_date"),   errors="coerce")
        if pd.isna(start_date) or pd.isna(end_date):
            return json_nocache({"error": "Invalid or missing start_date/end_date (use YYYY-MM-DD)"}, 400)
        if start_date > end_date:
            return json_nocache({"error": "start_date must be <= end_date"}, 400)

        sym1 = (p.get("sym1") or "").strip().upper()
        sym2 = (p.get("sym2") or "").strip().upper()
        if not sym1 or not sym2:
            return json_nocache({"error": "Please provide both sym1 and sym2"}, 400)

        model = get_model(dataset)
        d = get_df(dataset)
        if "Name" not in d.columns:
            return json_nocache({"error": f"{dataset} dataset has no 'Name' column"}, 400)

        def prep(symbol):
            ds = d[d["Name"].astype(str).str.upper() == symbol].copy()
            ds = ds[(ds["date"] >= start_date) & (ds["date"] <= end_date)]
            if ds.empty:
                return None
            X = ds[REQUIRED_FEATURES].to_numpy(dtype=float)
            y = ds["close"].to_numpy(dtype=float)
            yhat = model.predict(X)
            return {
                "dates": ds["date"].tolist(),
                "y_true": y.tolist(),
                "y_pred": yhat.tolist(),
                "metrics": {
                    "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
                    "mae":  float(mean_absolute_error(y, yhat)),
                    "r2":   float(r2_score(y, yhat)),
                    "mape": safe_mape(y, yhat),
                }
            }

        d1 = prep(sym1)
        d2 = prep(sym2)
        missing = [s for s,dx in ((sym1,d1),(sym2,d2)) if dx is None]
        if missing:
            return json_nocache({"error": f"No data for symbol(s) {', '.join(missing)} in this range"}, 404)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(d1["dates"], d1["y_true"], label=f"{sym1} Actual")
        plt.plot(d1["dates"], d1["y_pred"], label=f"{sym1} Predicted", linestyle="--")
        plt.legend(); plt.xticks(rotation=45)

        plt.subplot(2, 1, 2)
        plt.plot(d2["dates"], d2["y_true"], label=f"{sym2} Actual")
        plt.plot(d2["dates"], d2["y_pred"], label=f"{sym2} Predicted", linestyle="--")
        plt.legend(); plt.xticks(rotation=45)

        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150); buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8"); plt.close()

        return json_nocache({"graph": img_b64,
                             "metrics": {sym1: d1["metrics"], sym2: d2["metrics"]}})
    except Exception:
        logger.exception("Compare failed")
        return safe_error("Compare failed")
    
# ---------------------------
# Compare Multiple Models
# ---------------------------


@app.route("/compare_models", methods=["POST"])
def compare_models():
    try:
        payload = request.get_json()
        dataset = payload.get("dataset")
        start_str = payload.get("start_date")
        end_str = payload.get("end_date")
        sym = (payload.get("symbol") or "").strip().upper()

        if not dataset or not start_str or not end_str:
            return jsonify({"error": "Missing required parameters"}), 400

        # Dataset load
        path = os.path.join("datasets", f"{dataset}.csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

        # symbol column check
        col = "Name" if "Name" in df.columns else "symbol"

        mask = (
            (df["date"] >= pd.to_datetime(start_str)) &
            (df["date"] <= pd.to_datetime(end_str))
        )
        if sym:  # only filter symbol if provided
            mask &= (df[col].astype(str).str.upper() == sym)

        subset = df.loc[mask].copy()
        if subset.empty:
            return jsonify({"error": "No data for given range"}), 400

        X = subset[["open", "high", "low", "volume"]]
        y = subset["close"]

        # Models to compare
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Tuned Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
        }

        results = {}
        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)

            rmse = float(np.sqrt(mean_squared_error(y, preds)))
            mae = float(mean_absolute_error(y, preds))
            r2 = float(r2_score(y, preds))
            mape = safe_mape(y, preds)

            results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

        return jsonify({"metrics": results})

    except Exception as e:
        logging.error(f"Compare Models Error: {e}")
        return jsonify({"error": str(e)}), 500

# -------- CSV upload & predict ---------
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    try:
        dataset = get_dataset_key_from_request()
        model = get_model(dataset)

        f = request.files.get("file")
        if f is None or f.filename == "":
            return json_nocache({"error": "Attach a CSV file"}, 400)
        user_df = pd.read_csv(f)

        for c in REQUIRED_FEATURES:
            if c not in user_df.columns:
                return json_nocache({"error": f"Missing {c}"}, 400)

        for c in REQUIRED_FEATURES + (["close"] if "close" in user_df.columns else []):
            if c in user_df.columns:
                user_df[c] = pd.to_numeric(user_df[c], errors="coerce")

        clean_df = user_df.dropna(subset=REQUIRED_FEATURES).copy()
        X = clean_df[REQUIRED_FEATURES].to_numpy(dtype=float)
        preds = model.predict(X)
        clean_df["pred_close"] = preds

        metrics = {}
        if "close" in clean_df.columns:
            y_true = clean_df["close"].to_numpy(dtype=float)
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
                "mae": float(mean_absolute_error(y_true, preds)),
                "r2": float(r2_score(y_true, preds)),
                "mape": safe_mape(y_true, preds),
            }

        # CSV to Base64
        import base64, io
        csv_buffer = io.StringIO()
        clean_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')

        preview_rows = clean_df.head(10).to_dict(orient="records")

        return json_nocache({
            "rows_in": len(user_df),
            "rows_used": len(clean_df),
            "preview": preview_rows,
            "csv_b64": csv_b64,
            **metrics
        })
    except Exception:
        return safe_error("CSV upload failed")


# ---------------------------------------------------
if __name__ == "__main__":
    # app.run(host="127.0.0.1", port=5000, debug=False)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
