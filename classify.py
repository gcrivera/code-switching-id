from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

def fit(X, Y, save_path):
    gmm = GaussianMixture(4).fit(X, Y)
    joblib.dump(gmm, save_path)
    return

def predict(X, model):
    return model.predict(X)
