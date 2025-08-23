from flask import Flask, request, jsonify, render_template
import joblib, numpy as np

app = Flask(__name__)
model = joblib.load("random_forest_model_small.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)  # force handles wrong headers
        for k in ["open","high","low","volume"]:
            if k not in data:
                return jsonify({"error": f"Missing key: {k}"}), 400

        feats = np.array([[float(data["open"]), float(data["high"]),
                           float(data["low"]),  float(data["volume"])]])
        pred = float(model.predict(feats)[0])
        return jsonify({"prediction": pred})
    except Exception as e:
        # Always return JSON on error so frontend can show message
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)