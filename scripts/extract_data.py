import librosa
import numpy as np
import os
from tqdm import tqdm

def load_train():
    features = {'m': [], 'f': []}
    msa_path = '/data/sls/qcri/asr/data/vardial/vardial2017/train.vardial2017/wav/MSA/'
    egy_path = '/data/sls/qcri/asr/data/vardial/vardial2017/train.vardial2017/wav/EGY/'

    print 'Loading MSA training data...'
    for file in tqdm(os.listdir(msa_path)):
        if file.endswith('.wav'):
            wav_file = os.path.join(msa_path, file)
            x , sr = librosa.core.load(wav_file, sr=16000, mono=True, dtype='float')
            mfcc = librosa.feature.mfcc(x, sr, n_fft=400, hop_length=160, fmin=133, fmax=6955)
            width = mfcc.shape[1]
            if width % 2 == 0:
                width -= 1
            mfcc_delta = librosa.feature.delta(mfcc, width=width)
            mfcc_delta_delta = librosa.feature.delta(mfcc, width=width, order=2)

            X = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))
            X = np.matrix(X).T
            X = cmvn_slide(X, cmvn='m')
            if len(features['m']) == 0:
                features['m'] = X
            else:
                features['m'] = np.concatenate((features['m'], X))
            if len(features['m']) > 1000:
                break

    print 'Loading EGY training data...'
    for file in tqdm(os.listdir(egy_path)):
        if file.endswith('.wav'):
            wav_file = os.path.join(egy_path, file)
            x , sr = librosa.core.load(wav_file, sr=16000, mono=True, dtype='float')
            mfcc = librosa.feature.mfcc(x, sr, n_fft=400, hop_length=160, fmin=133, fmax=6955)
            width = mfcc.shape[1]
            if width % 2 == 0:
                width -= 1
            mfcc_delta = librosa.feature.delta(mfcc, width=width)
            mfcc_delta_delta = librosa.feature.delta(mfcc, width=width, order=2)

            X = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))
            X = np.matrix(X).T
            X = cmvn_slide(X, cmvn='m')
            if len(features['f']) == 0:
                features['f'] = X
            else:
                features['f'] = np.concatenate((features['f'], X))
            if len(features['f']) > 1000:
                break

    msa_features = open('../data/msa.npy')
    egy_features = open('../data/egy.npy')
    np.save(msa_features, features['m'])
    np.save(egy_features, features['f'])
    return

def cmvn_slide(X, win_len=300, cmvn=False):
    max_length = np.shape(X)[0]
    new_feat = np.empty_like(X)
    cur = 1
    left_win = 0
    right_win = win_len/2

    for cur in range(max_length):
        cur_slide = X[cur-left_win:cur+right_win,:]
        mean = np.mean(cur_slide,axis=0)
        std = np.std(cur_slide,axis=0)
        if cmvn == 'mv':
            new_feat[cur,:] = (X[cur,:]-mean)/std # for cmvn
        elif cmvn == 'm':
            new_feat[cur,:] = (X[cur,:]-mean) # for cmn
        if left_win < win_len/2:
            left_win += 1
        elif max_length-cur < win_len/2:
            right_win -= 1
    return new_feat

if __name__ == '__main__':
    load_train()