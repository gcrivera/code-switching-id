import bob.learn.em as em
import numpy as np
from bob.io.base import HDF5File
# from sklearn.base import clone
# from sklearn.externals import joblib
# from sklearn.mixture import GaussianMixture

def fit_ubm(data, save_path):
    print('Generating UBM...')
    all_data = np.concatenate((data['m'], data['f']))
    ubm = em.GMMMachine(64, 60)
    ubm_trainer = em.ML_GMMTrainer(True, True, True)
    em.train(ubm_trainer, ubm, all_data, max_iterations=12,
                convergence_threshold=0.001)
    ubm.save(HDF5File(save_path + 'ubm.h5', 'w'))
    # ubm = GaussianMixture(128, covariance_type='diag', init_params='random', warm_start=True, max_iter=12, verbose=1).fit(all_data)
    # joblib.dump(ubm, save_path + 'ubm.joblib')
    return ubm

def fit_adap(data, ubm, save_path):
    print('Generating GMMs...')
    msa = em.GMMMachine(64, 60)
    egy = em.GMMMachine(64, 60)
    msa_trainer = em.MAP_GMMTrainer(ubm, relevance_factor=10)
    egy_trainer = em.MAP_GMMTrainer(ubm, relevance_factor=10)
    em.train(msa_trainer, msa, data['f'], max_iterations=14,
                convergence_threshold=0.001)
    em.train(egy_trainer, egy, data['m'], max_iterations=14,
                convergence_threshold=0.001)
    msa.save(HDF5File(save_path + 'msa.h5', 'w'))
    egy.save(HDF5File(save_path + 'egy.h5', 'w'))
    # ubm.set_params(max_iter=14)
    # msa = clone(ubm).fit(data['m'])
    # egy = clone(ubm).fit(data['f'])
    # joblib.dump(msa, save_path + 'msa.joblib')
    # joblib.dump(egy, save_path + 'egy.joblib')

def fit(data, save_path):
    print('Generating GMMs...')
    msa = GaussianMixture(128, covariance_type='diag', init_params='random', n_init=5, max_iter=14, verbose=1).fit(data['m'])
    print('Finished MSA GMM...')
    egy = GaussianMixture(128, covariance_type='diag', init_params='random', n_init=5, max_iter=14, verbose=1).fit(data['f'])
    print('Finished EGY GMM...')
    joblib.dump(msa, save_path + 'msa.joblib')
    joblib.dump(egy, save_path + 'egy.joblib')
    return

def predict_ubm(data, model_path):
    try:
        msa = em.GMMMachine(HDF5File(model_path + 'msa.h5'))
        egy = em.GMMMachine(HDF5File(model_path + 'egy.h5'))
        # ubm = joblib.load(model_path + 'ubm.joblib')
        # msa = joblib.load(model_path + 'msa.joblib')
        # egy = joblib.load(model_path + 'egy.joblib')
    except :
        print("Model not found."); exit()

    return {'m': map(lambda x: (msa.log_likelihood(x), egy.log_likelihood(x)), data['m']),
            'f': map(lambda x: (msa.log_likelihood(x), egy.log_likelihood(x)), data['f'])}

def predict(data, model_path):
    try:
        msa = joblib.load(model_path + 'msa.joblib')
        egy = joblib.load(model_path + 'egy.joblib')
    except :
        print("Model not found."); exit()

    return {'m': map(lambda x: (msa.bic(x), egy.bic(x)), data['m']),
            'f': map(lambda x: (msa.bic(x), egy.bic(x)), data['f'])}


def select_models(data):
    try:
        msa_64_39 = joblib.load('models/64c_39f_25w_10s/msa.joblib')
        egy_64_39 = joblib.load('models/64c_39f_25w_10s/egy.joblib')
        msa_128_39 = joblib.load('models/128c_39f_25w_10s/msa.joblib')
        egy_128_39 = joblib.load('models/128c_39f_25w_10s/egy.joblib')
        msa_64_60 = joblib.load('models/64c_60f_25w_10s/msa.joblib')
        egy_64_60 = joblib.load('models/64c_60f_25w_10s/egy.joblib')
        msa_128_60 = joblib.load('models/128c_60f_25w_10s/msa.joblib')
        egy_128_60 = joblib.load('models/128c_60f_25w_10s/egy.joblib')
    except :
        print("Model not found."); exit()

    msa_data = np.concatenate(data['m'], axis=0)
    egy_data = np.concatenate(data['f'], axis=0)


    # print 'LL for MSA data on msa_64_39'
    # print msa_64_39.bic(msa_data)
    # print 'LL for MSA data on egy_64_39'
    # print egy_64_39.bic(msa_data)
    # print 'LL for MSA data on msa_128_39'
    # print msa_128_39.bic(msa_data)
    # print 'LL for MSA data on egy_128_39'
    # print egy_128_39.bic(msa_data)
    #
    # print '\n\n\n'
    #
    # print 'LL for EGY data on egy_64_39'
    # print egy_64_39.bic(egy_data)
    # print 'LL for EGY data on msa_64_39'
    # print msa_64_39.bic(egy_data)
    # print 'LL for EGY data on egy_128_39'
    # print egy_128_39.bic(egy_data)
    # print 'LL for EGY data on msa_128_39'
    # print msa_128_39.bic(egy_data)

    exit()
