from flask import Blueprint, request, render_template
import numpy as np
import pickle
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.DEBUG)

yield_prediction_app = Blueprint('yield_prediction_app', __name__)

# lazy model holders
dtr = None
preprocessor = None

def load_yield_models():
    global dtr, preprocessor
    if dtr is not None and preprocessor is not None:
        return True
    try:
        dtr_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'dtr.pkl')
        pre_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'preprocessor.pkl')
        if os.path.exists(dtr_path):
            with open(dtr_path, 'rb') as f:
                dtr = pickle.load(f)
        if os.path.exists(pre_path):
            with open(pre_path, 'rb') as f:
                preprocessor = pickle.load(f)
        if dtr is None or preprocessor is None:
            logging.error('Yield model or preprocessor not found in Data/.')
            return False
        return True
    except Exception as e:
        logging.error(f'Error loading yield models lazily: {e}')
        return False

@yield_prediction_app.route('/')
def index():
    return render_template('predict.html', prediction=None)

@yield_prediction_app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            crop = request.form['crop']
            crop_year = int(request.form['crop_year'])
            annual_rainfall = float(request.form['annual_rainfall'])
            pesticide = float(request.form['pesticide'])
            fertilizer = float(request.form['fertilizer'])
            area = float(request.form['area'])
            state = request.form['state']
            season = request.form['season']

            features = pd.DataFrame([[crop, crop_year, season, state, area, annual_rainfall, fertilizer, pesticide]],
                                    columns=['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

            # ensure models are loaded lazily
            if not load_yield_models():
                return render_template('predict.html', error='Model not loaded. Please train models first.')

            transformed_features = preprocessor.transform(features)

            prediction = dtr.predict(transformed_features)

            return render_template('predict.html', 
                                   prediction=f"Predicted Yield: {prediction[0]:.2f}")

        except ValueError as e:
            logging.error(f"Error: {e}")
            return render_template('predict.html', error=f"Input Error: {e}")
