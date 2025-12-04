from flask import Flask, request, render_template, Blueprint
import pickle
import pandas as pd

app = Flask(__name__, template_folder='../templates', static_folder='../static')
crop_recommendation_app = Blueprint('crop_recommendation_app', __name__)

try:
    with open('data/crop_model.pkl', 'rb') as f:
        crop_model = pickle.load(f)

    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model or label encoder: {e}")

@crop_recommendation_app.route('/')
def home():
    return render_template('recommend.html')

@crop_recommendation_app.route('/recommend', methods=['POST'])
def recommend():
    try:
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_value = float(request.form['ph_value'])
        rainfall = float(request.form['rainfall'])

        input_data = pd.DataFrame([{
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Temperature': temperature,
            'Humidity': humidity,
            'pH_Value': ph_value,
            'Rainfall': rainfall
        }])

        prediction_encoded = crop_model.predict(input_data)[0]
        crop_prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        return render_template('recommendation.html', recommendation=crop_prediction)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('recommend.html', error="Error during prediction. Please check your inputs.")

app.register_blueprint(crop_recommendation_app)


