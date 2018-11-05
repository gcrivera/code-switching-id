from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

def fit(data, save_path):
    print 'Generating GMMs...'
    msa = GaussianMixture(64, covariance_type='diag', init_params='random', n_init=5, verbose=1).fit(data['m'])
    print 'Finished MSA GMM...'
    egy = GaussianMixture(64, covariance_type='diag', init_params='random', n_init=5, verbose=1).fit(data['f'])
    print 'Finished EGY GMM...'
    joblib.dump(msa, save_path + 'msa.joblib')
    joblib.dump(egy, save_path + 'egy.joblib')
    return

def predict(data, model_path):
    try:
        msa = joblib.load(model_path + 'msa.joblib')
        egy = joblib.load(model_path + 'egy.joblib')
    except :
        print("Model not found."); exit()

    return {'m': map(lambda x: (msa.score(x), egy.score(x)), data['m']),
            'f': map(lambda x: (msa.score(x), egy.score(x)), data['f'])}
