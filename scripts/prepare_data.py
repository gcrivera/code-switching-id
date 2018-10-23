import librosa
import numpy as np

# Input: waveforms and word alignments
# Output: [X, Y] where X is MFCC features and Y is the tag 0-3

# m -> 0
# f -> 1
# fm -> 2
# g -> 3

def load(dataset):
    transcription = open('code-switching-stats-analysis/' + dataset + '_txt/all.txt.bw')
    lines = transcription.readlines()
    transcription.close()

    alignments = load_word_alignments()
    tags_dict = {'m': 0, 'f': 1, 'fm': 2, 'g': 3}
    features = []
    tags = []
    total_missing = 0

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
            total_missing += 1
            print 'Missing alignment: ' + wav_file
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
            X = librosa.feature.mfcc(x, sr, n_fft=512, hop_length=160, n_mfcc=40, fmin=133, fmax=6955)
            features.append(X)
            tags.append(tags_dict[word_tag[1]])

    print 'Total missing alignments: ' + str(total_missing)
    print len(features[0][0])
    return np.array(features),np.array(tags)


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
            alignments[curr_file] = curr_word_time_list
            curr_file = data[0]
            curr_word_time_list = []
        curr_word_time_list.append((data[4], (data[2], data[3])))

    alignments[curr_file] = curr_word_time_list

    return alignments