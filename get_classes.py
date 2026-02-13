import pickle
import os

le_path = os.path.join('Data', 'label_encoder.pkl')
try:
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
        print("Classes:", le.classes_)
except Exception as e:
    print(f"Error: {e}")
