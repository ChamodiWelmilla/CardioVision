from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models safely
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
except Exception as e:
    scaler = None
    kmeans = None
    print(f"Error loading models: {e}. Make sure to run train_model.py first.")

# Health advice based on typical cluster profiling (mockup examples for cardio value)
CLUSTER_ADVICE = {
    0: {
        "title": "Low Risk / Healthy Profile",
        "message": "Your vitals align with a generally healthy cardiovascular profile. Maintain your current balanced diet and regular exercise routine.",
        "color": "#10b981" # Emerald
    },
    1: {
        "title": "Elevated Blood Pressure",
        "message": "Your profile indicates elevated blood pressure. Consider reducing sodium intake, managing stress, and consulting a physician for monitoring.",
        "color": "#f59e0b" # Amber
    },
    2: {
        "title": "High Cholesterol Focus",
        "message": "Your readings show elevated cholesterol levels. Focus on a heart-healthy diet rich in omega-3s and fiber. Limit saturated fats.",
        "color": "#f97316" # Orange
    },
    3: {
        "title": "Elevated Heart Rate",
        "message": "Your maximum heart rate is higher than average for this cluster. Ensure you are well-hydrated and monitor your cardiovascular load.",
        "color": "#3b82f6" # Blue
    },
    4: {
        "title": "High Risk Profile",
        "message": "Multiple indicators (BP/Cholesterol) are elevated. We strongly recommend scheduling a comprehensive cardiovascular check-up with your doctor.",
        "color": "#ef4444" # Red
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not scaler or not kmeans:
        return jsonify({"error": "Models not loaded on the server."}), 500
        
    try:
        data = request.json
        age = float(data.get('age', 0))
        sex = float(data.get('sex', 0))
        bp = float(data.get('bloodPressure', 0))
        chol = float(data.get('cholesterol', 0))
        maxhr = float(data.get('maxHR', 0))
        
        # Prepare input array
        X_input = np.array([[age, sex, bp, chol, maxhr]])
        
        # Scale
        X_scaled = scaler.transform(X_input)
        
        # Predict cluster
        cluster_id = int(kmeans.predict(X_scaled)[0])
        
        advice = CLUSTER_ADVICE.get(cluster_id, {
            "title": "General Profile",
            "message": "Consult your physician for personalized medical advice.",
            "color": "#6b7280"
        })
        
        return jsonify({
            "cluster": cluster_id,
            "title": advice['title'],
            "message": advice['message'],
            "color": advice['color']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
