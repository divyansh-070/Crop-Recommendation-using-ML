import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

df = pd.read_csv("Data/Crop_Yield.csv")

col = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
df = df[col]

df.ffill(inplace=True)

X = df.drop('Yield', axis=1)
y = df['Yield']

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') 
scale = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('OneHotEncode', ohe, ['Crop', 'Season', 'State']),  
        ('StandardScale', scale, ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])  
    ],
    remainder='passthrough'  
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train_processed, y_train)

y_pred = dtr.predict(X_test_processed)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
model_accuracy = r2 * 100  

print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Model Accuracy: {model_accuracy:.2f}%")


if not os.path.exists("Data"):
    os.makedirs("Data")

pickle.dump(dtr, open("Data/dtr.pkl", "wb"))
pickle.dump(preprocessor, open("Data/preprocessor.pkl", "wb"))

print("Model and preprocessor saved successfully in the 'Data' directory.")
