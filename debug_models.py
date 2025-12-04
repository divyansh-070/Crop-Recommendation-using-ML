import pickle
import sys
import os

print("Python version:", sys.version)
try:
    import sklearn
    print("Scikit-learn version:", sklearn.__version__)
except ImportError:
    print("Scikit-learn not installed")

print("\n--- Testing Crop Recommendation Models ---")
try:
    with open('Data/crop_model.pkl', 'rb') as f:
        pickle.load(f)
    print("crop_model.pkl loaded successfully")
except Exception as e:
    print(f"Failed to load crop_model.pkl: {e}")

try:
    with open('Data/label_encoder.pkl', 'rb') as f:
        pickle.load(f)
    print("label_encoder.pkl loaded successfully")
except Exception as e:
    print(f"Failed to load label_encoder.pkl: {e}")

print("\n--- Testing Yield Prediction Models ---")
try:
    with open('Data/dtr.pkl', 'rb') as f:
        pickle.load(f)
    print("dtr.pkl loaded successfully")
except Exception as e:
    print(f"Failed to load dtr.pkl: {e}")

try:
    with open('Data/preprocessor.pkl', 'rb') as f:
        pickle.load(f)
    print("preprocessor.pkl loaded successfully")
except Exception as e:
    print(f"Failed to load preprocessor.pkl: {e}")
