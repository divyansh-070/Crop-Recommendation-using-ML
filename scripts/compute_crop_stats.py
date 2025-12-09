import pandas as pd, json, sys
cols=['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Rainfall']
try:
    df=pd.read_csv('Data/Crop_Recommendation.csv')
except Exception as e:
    print('ERROR', e)
    sys.exit(1)
stats={}
for c in cols:
    col=df[c].dropna().astype(float)
    stats[c]={'min':float(col.min()),'p1':float(col.quantile(0.01)),'mean':float(col.mean()),'std':float(col.std()),'p99':float(col.quantile(0.99)),'max':float(col.max())}
print(json.dumps(stats, indent=2))
