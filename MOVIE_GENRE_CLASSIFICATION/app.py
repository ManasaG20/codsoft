from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Load trained model
model = joblib.load("movie.joblib")

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# Predict genre
@app.route('/predict', methods=['POST'])
def predict_genre():
    data = request.get_json()
    plot = data.get("plot", "")
    
    if not plot.strip():
        return jsonify({"error": "No plot provided"}), 400

    pred = model.predict([plot])[0]
    
    # Optionally, also give probabilities if classifier supports it
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([plot])[0]
        confidence = round(max(probs)*100, 2)
    else:
        confidence = None

    return jsonify({
        "genre": pred,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
