import librosa

# Input: waveforms and word alignments
# Output: [X, Y] where X is MFCC features and Y is the tag 0-3

# m -> 0
# f -> 1
# fm -> 2
# g -> 3

def load(dataset):
    if dataset == 'train':
        # load train
        return
    elif dataset == 'val':
        # load val
        return
    elif dataset == 'test':
        # load test
        return
    else:
        return
    # load per word alignment
    x , sr = librosa.core.load(wavname, sr=16000, mono=True, offset=start, duration=end-start, dtype='float')
    X = librosa.feature.mfcc(x, sr, n_fft=512, hop_length=160, n_mfcc=40, fmin=133, fmax=6955)