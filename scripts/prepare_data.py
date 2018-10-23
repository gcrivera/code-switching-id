import librosa
import numpy as np

# Input: waveforms and word alignments
# Output: [X, Y] where X is MFCC features and Y is the tag 0-3

# m -> 0
# f -> 1
# fm -> 2
# g -> 3

def load(dataset):
    transcription = open('../code-switching-stats-analysis/' + dataset + '_txt/all.txt.bw')

    alignments = load_word_alignments()
    tags_dict = ['m': 0, 'f': 1, 'fm': 2, 'g': 3]
    features = []
    tags = []

    for line in transcription_lines.readlines():
        line_data = line.split(' ')
        wav_file = line_data[0]
        words = line_data[1:]
        line_alignments = alignments[wav_file]
        for i in range(len(words)):
            word_tag = words[i].split(' ')
            if len(word_tag) < 2:
                continue
            word_alignment = line_alignments[i]
            if word_tag[0] != word_alignment[0]:
                continue
            start = float(wav_file.split('_')[-2]) + float(word_alignment[1][0])
            x , sr = librosa.core.load(wav_file, sr=16000, mono=True, offset=start, duration=word_alignment[1][1], dtype='float')
            X = librosa.feature.mfcc(x, sr, n_fft=512, hop_length=160, n_mfcc=40, fmin=133, fmax=6955)
            features.append(X)
            tags.append(tags_dict[word_tag[1]])

    return np.array(features),np.array(tags)


def load_word_alignments():
    word_alignments = open('../code-switching-stats-analysis/egy_test_word_align.ctm')
    lines = word_alignments.readlines()
    word_alignments.close()

    data = lines[0].split(' ')
    alignments = {}
    curr_file = data[0]
    curr_word_time_list = [(data[4], (data[2], data[3]))]
    for line in word_alignments.readlines():
        data = line.split(' ')
        if data[0] != curr_file:
            alignments[curr_file] = curr_word_time_list
            curr_file = data[0]
            curr_word_time_list = []
        curr_word_time_list.append((data[4], (data[2], data[3])))

    alignments[curr_file] = curr_word_time_list

    return alignments