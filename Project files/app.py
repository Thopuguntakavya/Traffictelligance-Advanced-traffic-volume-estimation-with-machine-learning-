
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load model and scaler
        # import os
        base_path = os.path.dirname(__file__)
        model = pickle.load(open(os.path.join(base_path,'model.pkl'),'rb'))
        scaler = pickle.load(open(os.path.join(base_path,'scaler.pkl'),'rb'))
        # Collect inputs from form
        holiday = int(request.form['holiday'])
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = int(request.form['weather'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Create DataFrame from inputs
        input_dict = {
            'holiday': [holiday],
            'temp': [temp],
            'rain': [rain],
            'snow': [snow],
            'weather': [weather],
            'year': [year],
            'month': [month],
            'day': [day],
            'hours': [hours],
            'minutes': [minutes],
            'seconds': [seconds]
        }
        final_features = pd.DataFrame(input_dict)

        # REORDER columns to match scaler expectation
        expected_order = [
            'holiday', 'temp', 'rain', 'snow', 'weather', 
            'day','month','year', 'hours', 'minutes', 'seconds'
        ]
        final_features = final_features[expected_order]

        # Scale inputs
        final_features_scaled = scaler.transform(final_features)

        # Predict
        prediction = model.predict(final_features_scaled)[0]

        return render_template('output.html', title='Traffic Volume Estimation', result=f"Estimated Traffic Volume: {int(prediction)}")

if __name__ == "__main__":
    app.run(debug=True)
