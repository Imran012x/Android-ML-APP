from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

# Load your pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Define the root endpoint
@app.route('/')
def index():
    return "Welcome to this BOUSST Project"

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form.get('cgpa'))
    iq = float(request.form.get('iq'))
    profile_score = float(request.form.get('profile_score'))

    input_query = np.array([[cgpa, iq, profile_score]])

    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment variable
    app.run(host='0.0.0.0', port=port, debug=True)
