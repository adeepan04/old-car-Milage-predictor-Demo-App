"""Train an MPG predictor on auto-mpg.csv and export it (joblib + JSON for the web app)."""
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FEATURES = ['cylinders','displacement','horsepower','weight','acceleration','model_year',
            'log_weight','log_hp','log_disp','origin_2','origin_3']

def load(path='auto-mpg.csv'):
    df = pd.read_csv(path)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')   # '?' -> NaN
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
    return df

def make_features(cyl, disp, hp, wt, acc, year, origin):
    """Works on scalars or pandas Series."""
    return pd.DataFrame({
        'cylinders': cyl, 'displacement': disp, 'horsepower': hp, 'weight': wt,
        'acceleration': acc, 'model_year': year,
        'log_weight': np.log(wt), 'log_hp': np.log(hp), 'log_disp': np.log(disp),
        'origin_2': (np.asarray(origin) == 2).astype(int),
        'origin_3': (np.asarray(origin) == 3).astype(int)})[FEATURES]

def new_model():
    return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                     subsample=0.8, random_state=42)

if __name__ == '__main__':
    df = load()
    X = make_features(df.cylinders, df.displacement, df.horsepower, df.weight,
                      df.acceleration, df['model year'], df.origin)
    y = df.mpg

    # 1) honest hold-out evaluation
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    p = new_model().fit(Xtr, ytr).predict(Xte)
    metrics = {'r2': r2_score(yte, p), 'mae': mean_absolute_error(yte, p),
               'rmse': mean_squared_error(yte, p) ** 0.5}
    print('Hold-out:', {k: round(v, 3) for k, v in metrics.items()})

    # 2) out-of-fold residuals -> empirical 80% prediction band
    oof = cross_val_predict(new_model(), X, y, cv=KFold(5, shuffle=True, random_state=42))
    lo, hi = np.quantile(y - oof, [0.10, 0.90])
    print(f'80% band: {lo:+.2f} .. {hi:+.2f} mpg')

    # 3) final model on all data
    model = new_model().fit(X, y)
    joblib.dump(model, 'mpg_model.joblib')

    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append({'f': t.feature.tolist(), 'th': np.round(t.threshold, 6).tolist(),
                      'l': t.children_left.tolist(), 'r': t.children_right.tolist(),
                      'v': np.round(t.value[:, 0, 0], 6).tolist()})
    out = {'features': FEATURES, 'init': float(model.init_.predict(X.iloc[:1])[0]),
           'lr': model.learning_rate, 'trees': trees, 'metrics': metrics,
           'band': [float(lo), float(hi)],
           'ranges': {c: [float(df[c].min()), float(df[c].max())] for c in
                      ['displacement','horsepower','weight','acceleration','model year']}}
    json.dump(out, open('model.json', 'w'), separators=(',', ':'))
    print('exported', len(trees), 'trees')
