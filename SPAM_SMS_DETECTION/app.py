from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os


app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

pipeline = joblib.load("spam.joblib")


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Predict
    pred = pipeline.predict([message])[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([message])[0][1]
    print(pred)
    return jsonify({
        "label": "spam" if pred == 1 else "ham",
        "confidence": round(float(proba), 4) if proba is not None else None
    })

# Run server
if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5000)
