import numpy as np
import pandas as pd
import os
import json

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'Data')
csv_path = os.path.join(data_dir, 'Crop_Recommendation.csv')
npz_path = os.path.join(data_dir, 'mahalanobis.npz')

cols=['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Rainfall']

df = pd.read_csv(csv_path)
X = df[cols].dropna().astype(float)

# compute mean and covariance; add small ridge for numerical stability
mu = X.mean().to_numpy()
cov = np.cov(X.values, rowvar=False)
# regularize covariance
eps = 1e-6 * np.eye(cov.shape[0])
cov_reg = cov + eps
inv_cov = np.linalg.inv(cov_reg)

# threshold for chi-square with dof=len(cols)
# try to use scipy if available; otherwise hardcode 0.99 quantile
try:
    from scipy.stats import chi2
    threshold = float(chi2.ppf(0.99, df=len(cols)))
except Exception:
    # df=7, chi2.ppf(0.99,7) ≈ 18.475
    threshold = 18.475

np.savez(npz_path, mu=mu, inv_cov=inv_cov, threshold=threshold)
print('Saved mahalanobis data to', npz_path)
print('threshold=', threshold)
print('mu=', mu.tolist())
