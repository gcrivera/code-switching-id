from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

# Best classifier: maxiter_250_ninit_5.joblib

def fit(X, save_path):
    print 'Fitting classifier...'

    gmm = GaussianMixture(4, max_iter=250, n_init=5).fit(X, Y)
    joblib.dump(gmm, save_path)
    return gmm

def predict(X, model):
    return model.predict(X)
