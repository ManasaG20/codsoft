from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Load model
model = joblib.load("customer_churn_model.joblib")

# Load feature info
df_sample = pd.read_csv("dataset.csv")
numeric_features = df_sample.select_dtypes(include=['int64','float64']).columns.drop(['Exited','RowNumber','CustomerId']).tolist()
categorical_features = df_sample.select_dtypes(include=['object']).columns.tolist()

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Convert to DataFrame
    df_input = pd.DataFrame([data])
    # Ensure all columns exist
    for col in numeric_features + categorical_features:
        if col not in df_input.columns:
            df_input[col] = 0  # default value

    # Predict
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]  # probability of churn
    return jsonify({"prediction": int(pred), "churn_prob": round(float(prob), 2)})

if __name__ == "__main__":
    app.run(debug=True)
