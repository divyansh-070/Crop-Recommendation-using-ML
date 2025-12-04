import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

data_dir = os.path.join(os.path.dirname(__file__), '../data')
df = pd.read_csv(os.path.join(data_dir, 'Crop_Recommendation.csv'))

df.columns = df.columns.str.strip()
print("Columns in the dataset:", df.columns)

expected_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Crop']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"The following columns are missing from the dataset: {missing_columns}")

X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
y = df['Crop']


le = LabelEncoder()
y_encoded = le.fit_transform(y)

with open(os.path.join(data_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open(os.path.join(data_dir, 'crop_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

print("Model training complete. Model saved as 'crop_model.pkl'")
