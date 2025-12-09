import pickle
import pandas as pd
m=pickle.load(open('Data/crop_model.pkl','rb'))
le=pickle.load(open('Data/label_encoder.pkl','rb'))
import numpy as np
extreme = pd.DataFrame([{'Nitrogen':1e6,'Phosphorus':1e6,'Potassium':1e6,'Temperature':1000.0,'Humidity':500.0,'pH_Value':100.0,'Rainfall':1e6}])
print('Input:')
print(extreme.to_dict(orient='records')[0])
pred = m.predict(extreme)
print('Encoded prediction:', pred)
print('Decoded prediction:', le.inverse_transform(pred))
