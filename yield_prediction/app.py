from flask import Blueprint, request, render_template
import numpy as np
import pickle
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

yield_prediction_app = Blueprint('yield_prediction_app', __name__)

dtr = pickle.load(open('Data/dtr.pkl', 'rb'))
preprocessor = pickle.load(open('Data/preprocessor.pkl', 'rb'))

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

            transformed_features = preprocessor.transform(features)

            prediction = dtr.predict(transformed_features)

            return render_template('predict.html', 
                                   prediction=f"Predicted Yield: {prediction[0]:.2f}")

        except ValueError as e:
            logging.error(f"Error: {e}")
            return render_template('predict.html', error=f"Input Error: {e}")
