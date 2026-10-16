
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "iris_model.pkl")

logging.basicConfig(
    filename=str(BASE_DIR / "ml_api.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris MLOps API is running",
        "endpoint": "/predict"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    features = data.get("features")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": "Provide exactly four numeric features"
        }), 400

    try:
        values = [float(x) for x in features]

        if not all(pd.notna(x) and abs(x) != float("inf")
                   for x in values):
            raise ValueError("Features must be finite numbers")

    except (ValueError, TypeError):
        return jsonify({
            "error": "All features must be finite numbers"
        }), 400

    input_df = pd.DataFrame([values], columns=FEATURES)
    prediction = str(model.predict(input_df)[0])

    logging.info({
        "input": values,
        "prediction": prediction
    })

    return jsonify({
        "prediction": prediction,
        "input_features": values
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
