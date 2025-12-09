"""
Validate normal vs extreme examples against dataset percentile checks and Mahalanobis stats.
Prints results for quick developer verification.
"""
import numpy as np
import pandas as pd
import os
import pickle

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base, 'Data')
csv_path = os.path.join(data_dir, 'Crop_Recommendation.csv')
npz_path = os.path.join(data_dir, 'mahalanobis.npz')
model_path = os.path.join(data_dir, 'crop_model.pkl')
le_path = os.path.join(data_dir, 'label_encoder.pkl')

cols = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Rainfall']

def load_stats():
    df = pd.read_csv(csv_path)
    stats = {}
    for c in cols:
        col = df[c].dropna().astype(float)
        stats[c] = {
            'p1': float(col.quantile(0.01)),
            'p99': float(col.quantile(0.99)),
            'min': float(col.min()),
            'max': float(col.max())
        }
    return stats


def percentile_check(stats, sample):
    bad = []
    for c in cols:
        v = sample[c]
        if v < stats[c]['p1'] or v > stats[c]['p99']:
            bad.append((c, v, stats[c]['p1'], stats[c]['p99']))
    return bad


def mahalanobis_check(sample):
    if not os.path.exists(npz_path):
        return None, 'mahalanobis file missing'
    d = np.load(npz_path)
    mu = d['mu']
    inv_cov = d['inv_cov']
    threshold = float(d['threshold'])
    x = np.array([sample[c] for c in cols])
    diff = x - mu
    m2 = float(diff.dot(inv_cov).dot(diff))
    return m2, threshold


if __name__ == '__main__':
    stats = load_stats()

    # pick a typical sample (use dataset first row)
    df = pd.read_csv(csv_path)
    typical = {c: float(df[c].iloc[0]) for c in cols}

    extreme = {c: 1e6 if c not in ['Temperature','pH_Value','Humidity'] else (1000.0 if c=='Temperature' else (100.0 if c=='pH_Value' else 500.0)) for c in cols}

    print('Typical sample:')
    print(typical)
    print('Percentile check (typical):', percentile_check(stats, typical))
    m2, thr = mahalanobis_check(typical)
    print(f'Mahalanobis (typical): {m2:.3f} threshold={thr:.3f}')

    print('\nExtreme sample:')
    print(extreme)
    print('Percentile check (extreme):', percentile_check(stats, extreme))
    m2e, thre = mahalanobis_check(extreme)
    print(f'Mahalanobis (extreme): {m2e:.3f} threshold={thre:.3f}')

    # show what the model predicts for each (for diagnostic only)
    try:
        m = pickle.load(open(model_path,'rb'))
        le = pickle.load(open(le_path,'rb'))
        import pandas as pd
        tdf = pd.DataFrame([typical])
        edf = pd.DataFrame([extreme])
        pt = m.predict(tdf)
        pe = m.predict(edf)
        print('\nModel predictions:')
        print('typical encoded ->', pt, 'decoded ->', le.inverse_transform(pt))
        print('extreme encoded ->', pe, 'decoded ->', le.inverse_transform(pe))
    except Exception as e:
        print('Model prediction check failed:', e)
