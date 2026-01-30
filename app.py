from flask import Flask, render_template, request, jsonify
from src.predict import predict_heart_disease

app = Flask(__name__)

# -------------------------
# HTML ROUTE
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_html():
    user_input = request.form['features']
    features = [float(x) for x in user_input.split(',')]

    result = predict_heart_disease(features)

    if result == 1:
        message = "HIGH RISK of Heart Disease"
    else:
        message = "LOW RISK of Heart Disease"

    return render_template('index.html', result=message)

# -------------------------
# API ROUTE (JSON)
# -------------------------
@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()

        # Check if JSON exists
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Check if 'features' key exists
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data["features"]

        # Check correct number of inputs
        if len(features) != 13:
            return jsonify({
                "error": "Exactly 13 feature values are required"
            }), 400

        # Convert all values to float
        features = [float(x) for x in features]

        result = predict_heart_disease(features)

        if result == 1:
            risk = "HIGH"
        else:
            risk = "LOW"

        return jsonify({
            "prediction": int(result),
            "risk_level": risk
        })

    except ValueError:
        return jsonify({
            "error": "All feature values must be numbers"
        }), 400

    except Exception as e:
        return jsonify({
            "error": "Something went wrong",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

