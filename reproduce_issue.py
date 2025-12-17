import pickle
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

data_dir = os.path.join(os.path.dirname(__file__), 'Data')
model_path = os.path.join(data_dir, 'crop_model.pkl')
le_path = os.path.join(data_dir, 'label_encoder.pkl')

print("Loading models...")
try:
    with open(model_path, 'rb') as f:
        crop_model = pickle.load(f)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

try:
    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded.")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    sys.exit(1)

# Dummy input
input_values = {
    'Nitrogen': 90,
    'Phosphorus': 42,
    'Potassium': 43,
    'Temperature': 20.8,
    'Humidity': 82.0,
    'pH_Value': 6.5,
    'Rainfall': 202.9
}

input_data = pd.DataFrame([input_values])
print("Input data shape:", input_data.shape)
print("Input columns:", input_data.columns.tolist())

with open('debug_log.txt', 'w') as log_file:
    def log(msg):
        print(msg)
        log_file.write(str(msg) + '\n')
    
    log("Attempting prediction...")
    try:
        proba = crop_model.predict_proba(input_data)[0]
        log(f"Probabilities shape: {proba.shape}")
        
        top3_indices = np.argsort(proba)[-3:][::-1]
        log(f"Top 3 indices: {top3_indices}")
        
        top3_crops = label_encoder.inverse_transform(top3_indices)
        log(f"Top 3 crops: {top3_crops}")
        
    except Exception as e:
        log("Error during prediction!")
        import traceback
        traceback.print_exc(file=log_file)
        traceback.print_exc()
