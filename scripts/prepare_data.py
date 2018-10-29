import librosa
import numpy as np
import os

# Input: waveforms and word alignments
# Output: dictionary mapping tag -> numpy array of extracted mfcc features

def load_train():
    features = {'m': [], 'f': []}
    msa_path = '/data/sls/qcri/asr/data/vardial/vardial2017/train.vardial2017/wav/MSA/'
    egy_path = '/data/sls/qcri/asr/data/vardial/vardial2017/train.vardial2017/wav/EGY/'

    for file in os.listdir(msa_path):
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
            if len(features['m']) == 0:
                features['m'] = X
            features['m'] = np.concatenate((features['m'], X), axis=1)

    for file in os.listdir(egy_path):
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
            if len(features['f']) == 0:
                features['f'] = X
            features['f'] = np.concatenate((features['f'], X), axis=1)

    return features

def load_test(dataset):
    print 'Loading ' + dataset + ' data...'

    transcription = open('code-switching-stats-analysis/' + dataset + '_txt/all.txt.bw')
    lines = transcription.readlines()
    transcription.close()

    alignments = load_word_alignments()
    features = {'m': [], 'f': [], 'fm': [], 'g': []}

    for line in lines:
        line = line.rstrip()
        line_data = line.split(' ')
        wav_file = line_data[0]
        wav_file_underscore = wav_file.split('_')
        utterance_start = wav_file_underscore[-2]
        wav_file_name = wav_file_underscore[0] + '__' + wav_file_underscore[1] + '_' + wav_file_underscore[2] + '.wav'
        wav_file_path = 'code-switching-stats-analysis/test_EGY_all/' + wav_file_name
        words = line_data[1:]
        try:
            line_alignments = alignments[wav_file]
        except:
            continue
        for i in range(len(words)):
            word_tag = words[i].split('_')
            if len(word_tag) < 2:
                continue
            word_alignment = line_alignments[i]
            if word_tag[0] != word_alignment[0]:
                continue
            start = float(utterance_start) + float(word_alignment[1][0])
            duration = float(word_alignment[1][1])

            x , sr = librosa.core.load(wav_file_path, sr=16000, mono=True, offset=start, duration=duration, dtype='float')
            mfcc = librosa.feature.mfcc(x, sr, n_fft=400, hop_length=160, fmin=133, fmax=6955)
            width = mfcc.shape[1]
            if width % 2 == 0:
                width -= 1
            mfcc_delta = librosa.feature.delta(mfcc, width=width)
            mfcc_delta_delta = librosa.feature.delta(mfcc, width=width, order=2)

            X = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))
            if len(features[word_tag[1]]) == 0:
                features[word_tag[1]] = X
            features[word_tag[1]] = np.concatenate((features[word_tag[1]], X), axis=1)

    return features


def load_word_alignments():
    word_alignments = open('code-switching-stats-analysis/egy_test_word_align.ctm')
    lines = word_alignments.readlines()
    word_alignments.close()

    data = lines[0].split(' ')
    alignments = {}
    curr_file = data[0]
    curr_word_time_list = [(data[4], (data[2], data[3]))]
    for line in lines[1:]:
        line = line.rstrip()
        data = line.split(' ')
        if data[0] != curr_file:
            curr_word_time_list = sorted(curr_word_time_list, key=lambda x: float(x[1][0]))
            alignments[curr_file] = curr_word_time_list
            curr_file = data[0]
            curr_word_time_list = []
        curr_word_time_list.append((data[4], (data[2], data[3])))

    curr_word_time_list = sorted(curr_word_time_list, key=lambda x: float(x[1][0]))
    alignments[curr_file] = curr_word_time_list

    return alignments