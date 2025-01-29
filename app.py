
from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'C:\Diabetes Prediction\Model\modelForPrediction.pkl'  # Replace with your model's path
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
SCALER_PATH = 'C:\Diabetes Prediction\Model\standard_scaler.pkl'  # Replace with your scaler's path
with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)

# Define the number of features expected by the model
EXPECTED_FEATURES = 8

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [float(x) for x in request.form.values()]

        # Validate the number of features
        if len(features) != EXPECTED_FEATURES:
            raise ValueError(f'Expected {EXPECTED_FEATURES} features, but received {len(features)}')

        # Scale the features
        final_features = scaler.transform(np.array(features).reshape(1, -1))

        # Make prediction
        prediction = model.predict(final_features)
        output = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

        return render_template('index.html', prediction_text=f'The person is {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
