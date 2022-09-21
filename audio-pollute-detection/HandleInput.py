import os, glob, time
import math, random
import numpy as np
import librosa, scipy
import matplotlib.pyplot as plt
# import matplotlib.image as img
from PIL import Image as img
from scipy.stats import kurtosis, skew, gmean
from sklearn import preprocessing


# save aufio features as csv
def save_chromagram(path, file_name, sample_rate, bins):
    # print('Extracting chromagram from {}{}.wav'.format(path, file_name))
    y, sr = librosa.load('{}{}.wav'.format(path, file_name), sr=sample_rate, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sample_rate, n_chroma=bins)
    chroma = np.transpose(chroma)
    np.savetxt('{}{}.csv'.format(path, file_name), chroma, delimiter=',')
    print('{} done'.format(file_name))

def save_mel_spectrogram(path, file_name, sample_rate, window_size, window_step, bands, fmin, fmax, pcen):
    # print('Extracting mel spectrogram from {}{}.wav'.format(path, file_name))
    y, sr = librosa.load('{}{}.wav'.format(path, file_name), sr=sample_rate, mono=True)
    if pcen:
        mel = librosa.feature.melspectrogram(y, sample_rate, n_fft=window_size, hop_length=window_step, power=1, n_mels=bands, fmin=fmin, fmax=fmax)
        mel = librosa.core.pcen(mel * (2**31), sr=sample_rate, hop_length=window_step)
        print('extractring pcen spectrogram for {}...'.format(file_name))
    else:
        mel = librosa.feature.melspectrogram(y, sample_rate, n_fft=window_size, hop_length=window_step, power=2, n_mels=bands, fmin=fmin, fmax=fmax)
        print('extractring mel spectrogram for {}...'.format(file_name))
    mel = np.transpose(mel)
    np.savetxt('{}{}.csv'.format(path, file_name), mel, fmt='%.12f', delimiter=',')

    print('done')

def save_stft_spectrogram(path, file_name, sample_rate, window_size, window_step):
    # print('Extracting spectrogram from {}{}.wav'.format(path, file_name))
    y, sr = librosa.load('{}{}.wav'.format(path, file_name), sr=sample_rate, mono=True)
    stft = librosa.stft(y, window_size, window_step)
    d, p = librosa.magphase(stft) #np.abs(stft)
    mel = librosa.feature.melspectrogram(S=stft)
    # librosa.feature.mfcc(S=mel, n_mfcc=12)
    # librosa.feature.spectral_centroid(S=d)
    # librosa.feature.spectral_bandwidth(S=d)
    # librosa.feature.spectral_contrast(S=d)
    # librosa.feature.spectral_flatness(S=d)
    # librosa.feature.spectral_rolloff(S=d)
    # librosa.feature.spectral_centroid(S=d)
    # librosa.feature.spectral_bandwidth(S=d)
    # librosa.feature.spectral_contrast(S=d)
    # librosa.feature.spectral_flatness(S=d)
    # librosa.feature.spectral_rolloff(S=d)
    d = np.transpose(d)
    np.savetxt('{}{}.csv'.format(path, file_name), d, delimiter=',')
    print('{} done'.format(file_name))

def save_stfv(path, file_name, sample_rate, window_size, window_step):
    # print('Extracting spectrogram from {}{}.wav'.format(path, file_name))
    y, sr = librosa.load('{}{}.wav'.format(path, file_name), sr=sample_rate, mono=True)
    stft = librosa.stft(y, window_size, window_step)
    d, p = librosa.magphase(stft) #np.abs(stft)
    mel = librosa.feature.melspectrogram(S=stft)

    header = 'MFCC00, MFCC01, MFCC02, MFCC03, MFCC04, MFCC05, MFCC06, MFCC07, MFCC08, MFCC09, MFCC10, MFCC11, MFCC12, MFCC13, CONT00, CONT01, COTN02, CONT03, CONT04, CONT05, CONT06, BANDWI, CENTRO, FLATNE, ROLLOF'
    f1, _ = librosa.magphase(librosa.feature.mfcc(S=mel, n_mfcc=14))
    f2 = librosa.feature.spectral_contrast(S=d)
    f3 = librosa.feature.spectral_bandwidth(S=d)
    f4 = librosa.feature.spectral_centroid(S=d)
    f5 = librosa.feature.spectral_flatness(S=d)
    f6 = librosa.feature.spectral_rolloff(S=d)
    stfv = np.concatenate((f1, f2, f3, f4, f5, f6), axis=0)
    # print(f1.shape, f2.shape, f3.shape, f4.shape, f5.shape, f6.shape)
    # print(stfv.shape)
    stfv = np.transpose(stfv)
    np.savetxt('{}{}.csv'.format(path, file_name), stfv, delimiter=',', header=header,  comments='')
    print('{} done'.format(file_name))


# load features from csv
def load_features_csv(path, category):

    # timing configuration
    sample_rate = 22050
    fft_size = 256*128
    if category == 0:
        #
        print('Total Window Length: {0:.3f}s'.format(fft_size/sample_rate))

    # load data
    raw = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)

    # clean data
    raw = np.where(np.isnan(raw), 0, raw)
    raw = np.where(np.isinf(raw), 0, raw)

    if raw.ndim < 2:
        raw = raw.reshape((raw.shape[0], 1))

    # fill output
    data = raw
    truth = np.full(int(raw.shape[0]), category)

    return data, truth

def load_features_csv_ts(path, category):

    ts_length = 64
    ts_step = 64

    print('Total Window Length: {0:.3f}s'.format(0.0116*ts_step))

    raw = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
    raw[:, 4] = np.log(raw[:, 4])
    raw[:, 3] = np.log(raw[:, 3])
    raw[:, 2] = np.log(raw[:, 2])
    raw = np.delete(raw, [0, 1, 2, 8, 10], axis=1)
    data = preprocessing.normalize(raw[0::2, :], norm='l2')

    ts_data = np.zeros((int(data.shape[0]/ts_step), ts_length, data.shape[1]))
    ts_truth = np.full(int(data.shape[0]/ts_step), category)
    for i in range(0, data.shape[0]-ts_length, ts_step):
        ts_data[int(i/ts_step), :, :] = data[i:i+ts_length, :]

    return ts_data, ts_truth


# Load audio from waveforms
def load_audio(path, category, normalize=0):

    # configuration
    dBFSThreshold = -96                 # in dBFS
    sample_rate = 22050                 # in Hertz
    length = int(sample_rate * 1.0)     # in seconds
    step = int(length * 0.5)            # in seconds
    persa_au = 3                        # +/- dB (1 to 3 :: 3)
    persa_snr = 9                       # dB below the mean energy of the sample (6 to 18 :: 10)
    log = False

    # prepare data
    noise, rate = librosa.load('D:\\PhD\\Datasets\\pink_noise_1h.wav', sr=sample_rate, mono=True)
    raw, rate = librosa.load(path, sr=sample_rate, mono=True)
    for i in range(0, raw.shape[0]-max(length,step), step):
        if i == 0: count = 0
        column = raw[i:i + length]
        dbFS = 10*np.log10(np.square(column).mean()) if np.square(column).mean() > 0 else -math.inf
        if dbFS > dBFSThreshold:
            count = count + 1

    # load data
    data = np.zeros((count, length))
    truth = np.full(count, category)
    for i in range(0, raw.shape[0]-max(length,step), step):
        if i == 0: count = 0
        column = raw[i:i + length]
        dbFS = 10 * np.log10(np.square(column).mean()) if np.square(column).mean() > 0 else -math.inf

        if dbFS > dBFSThreshold:

            # NONE
            if normalize == 0:
                #
                pass

            # AU
            if normalize == 1:
                #
                column = column * np.random.uniform(pow(10, -persa_au / 20.), pow(10, persa_au / 20.))

            # PERSA
            if normalize == 8:
                column = column[:] / np.sqrt(np.mean(np.square(column)))
                # column = column[:] / np.amax(np.abs(column))
                column = column * np.random.uniform(pow(10, -persa_au/20.), pow(10, persa_au/20.))

            # PERSA+
            if normalize == 9:
                n = noise[i%(noise.shape[0]-length):i%(noise.shape[0]-length)+length]

                p_s = 10 * np.log10(np.square(column).mean() + np.finfo(float).eps)
                p_n = 10 * np.log10(np.square(n).mean() + np.finfo(float).eps)
                n = n * pow(10, (p_s - p_n - persa_snr) / 20.)
                column = column + n
                # print('Signal PW: {:.2f} | Noise PW: {:.2f} | Noise PWC: {:.2f} '.format(p_s, p_n, 10*np.log10(np.square(n).mean())))

                column = column[:] / np.sqrt(np.mean(np.square(column)))
                column = column * np.random.uniform(pow(10, -persa_au / 20.), pow(10, persa_au / 20.))

            if log:
                sign = np.sign(column)
                column = np.log10(np.abs(column) + 1)
                column = np.multiply(column, sign)

            data[count, :] = column
            # plt.plot(column[500:600])
            # plt.show()

            count = count + 1

    print('Class: {}, Full Size: {}, Filtered Size: {}, Window Length: {:.2f}s, Step Length: {:.2f}s'.format(category, int(raw.shape[0]/step), data.shape[0], length/sample_rate, step/sample_rate))

    noise = None
    data = data.reshape(data.shape[0], data.shape[1], 1)
    truth = truth.reshape(truth.shape[0], 1)
    return data, truth

def load_audio_ts(path, category):

    sample_rate = 22050
    length = 1024
    step = 1024
    ts_length = 20
    ts_step = 20

    # texture formation
    raw, rate = librosa.load(path, sr=sample_rate, mono=True)
    if rate != sample_rate: raw = librosa.resample(raw, rate, sample_rate)
    data = np.zeros((int(raw.shape[0]/step), length))
    for i in range(0, raw.shape[0]-length, step):
        column = raw[i:i + length]
        # if np.amax(column) > 0:
        #     column = column[:] / np.amax(column)
        data[int(i/step), :] = column

    # embedding formation
    ts_data = np.zeros((int(raw.shape[0] / step / ts_step), ts_length, length))
    ts_truth = np.full(int(raw.shape[0] / step / ts_step), category)
    for i in range(0, int(raw.shape[0]/step)-ts_length, ts_step):
        rms = math.sqrt(np.square(data[i:i+ts_length, :]).mean())
        ts_data[int(i/ts_step), :, :] = data[i:i+ts_length, :] / rms

    print('Total Window Length: {0:.1f}s'.format(ts_length * step / rate))

    return ts_data, ts_truth


# Load spectrogram from csv
def load_spectrum_csv(path, category, normalize=0, poison_fold=-1, poison_type='none'):

    '''
    :param str path:
    :param int category: 0, 1, 2, ...
    :param int normalize: 0 NONE, 1 AU, 4 PCEN, 7 PONS, 8 PERSA, 9 PERSA+
    :param int poison_fold: -1, 0 or fold
    :param str poison_type: none, eq, snr
    :return: score
    :rtype: float
    '''


    # timing configuration
    sample_rate = 22050
    fft_step = 256
    length = int(sample_rate/fft_step * 1.0)     # in seconds
    step = int(length * 0.5)                     # in seconds... 0.25, 1.25


    # distortion configuration
    eq_snr = 10
    poison_snr = 9
    poison_std = 3


    # au configuration
    au = 30
    # pcen configuration
    pcen_au = 0                     # +/- dB (1 to 3 :: 1)
    # pons configuration
    pons_au = 3                     # +/- dB (1 to 3 :: 1)
    pons_snr = 30                   # dB below the max energy of the sample (24 to 42 :: 30)
    # persa & persa+ configuration
    persa_au = 3                    # +/- dB (1 to 3 :: 3)
    persa_snr = 9                   # dB below the mean energy of the sample (6 to 18 :: 10)
    persa_auto = False              # estimate the persa_snr parameter


    # pcen & persa+ preparation
    if normalize == 4:
        path = path.replace('MEL', 'PCEN')
        if category == 0:
            print('Changing path to {}\\'.format(path.rsplit('\\', 1)[0]))
    if normalize == 9:
        x, x_rate = librosa.load('D:\\PhD\\Datasets\\pink_noise_1h.wav', sr=sample_rate, mono=True)
        mel_n = librosa.feature.melspectrogram(x, x_rate, n_fft=512, hop_length=256, power=2.0, n_mels=56, fmin=100, fmax=8000)
        mel_n = np.transpose(mel_n)


    # texture formation
    raw = np.genfromtxt(path, delimiter=',')
    data = np.zeros((int(raw.shape[0]/step), length, raw.shape[1]))
    truth = np.full((int(raw.shape[0]/step), 1), category)
    metric = np.zeros((int(raw.shape[0]/step), 5))


    # eq degradation
    if poison_fold >= 0 and poison_type == 'eq':
        for j in range(eq_snr):
            for i in range(raw.shape[0]):
                if i < 1 * raw.shape[0] / 3:
                    if poison_fold == 3 or poison_fold == 0: raw[i, j] = raw[i, j] / pow(10, (eq_snr-j) / 10.)
                elif i < 2 * raw.shape[0] / 3:
                    if poison_fold == 2 or poison_fold == 0: raw[i, j] = raw[i, j] / pow(10, (eq_snr-j) / 10.)
                else:
                    if poison_fold == 1 or poison_fold == 0: raw[i, j] = raw[i, j] / pow(10, (eq_snr-j) / 10.)
    # snr degradation
    if poison_fold >= 0 and poison_type == 'snr':

        length = int(length*2)

        noise_a = np.genfromtxt('D:\\PhD\\Datasets\\LVLib-SMO-v1\\MEL\\music.csv', delimiter=',')
        noise_b = np.genfromtxt('D:\\PhD\\Datasets\\LVLib-SMO-v1\\MEL\\speech.csv', delimiter=',')
        noise_c = np.genfromtxt('D:\\PhD\\Datasets\\LVLib-SMO-v1\\MEL\\others.csv', delimiter=',')
        x, x_rate = librosa.load('D:\\PhD\\Datasets\\pump.wav', sr=sample_rate, mono=True)
        mel_nw = librosa.feature.melspectrogram(x, 22050, n_fft=512, hop_length=256, power=2.0, n_mels=56, fmin=100, fmax=8000)
        mel_nw = np.transpose(mel_nw)

        for i in range(0, raw.shape[0]-length, length):

            column = raw[i:i + length, :]

            if category == 0:
                if i < 1 * raw.shape[0] / 3:
                    n = noise_b[i:i + length, :]
                elif i < 2 * raw.shape[0] / 3:
                    n = noise_c[i:i + length, :]
                else:
                    n = mel_nw[i:i + length, :]
            elif category == 1:
                if i < 1 * raw.shape[0] / 3:
                    n = noise_a[i:i + length, :]
                elif i < 2 * raw.shape[0] / 3:
                    n = mel_nw[i:i + length, :]
                else:
                    n = noise_c[i:i + length, :]
            else:
                if i < 1 * raw.shape[0] / 3:
                    n = mel_nw[i:i + length, :]
                elif i < 2 * raw.shape[0] / 3:
                    n = noise_a[i:i + length, :]
                else:
                    n = noise_b[i:i + length, :]

            p_s = 10 * np.log10(np.mean(column))
            p_n = 10 * np.log10(np.mean(n))
            n = n * pow(10, (p_s - p_n - poison_snr + np.random.normal(0, poison_std)) / 10.)

            if category == 0:
                if i < 1 * raw.shape[0] / 3:
                    # n = n * pow(10, (p_s - p_n - poison_snr+poison_std) / 10.)
                    if poison_fold == 3 or poison_fold == 0: column = column + n
                elif i < 2 * raw.shape[0] / 3:
                    # n = n * pow(10, (p_s - p_n - poison_snr-poison_std) / 10.)
                    if poison_fold == 2 or poison_fold == 0: column = column + n
                else:
                    # n = n * pow(10, (p_s - p_n - poison_snr) / 10.)
                    if poison_fold == 1 or poison_fold == 0: column = column + n
            elif category == 1:
                if i < 1 * raw.shape[0] / 3:
                    # n = n * pow(10, (p_s - p_n - poison_snr) / 10.)
                    if poison_fold == 3 or poison_fold == 0: column = column + n
                elif i < 2 * raw.shape[0] / 3:
                    # n = n * pow(10, (p_s - p_n - poison_snr+poison_std) / 10.)
                    if poison_fold == 2 or poison_fold == 0: column = column + n
                else:
                    n = n * pow(10, (p_s - p_n - poison_snr-poison_std) / 10.)
                    if poison_fold == 1 or poison_fold == 0: column = column + n
            else:
                if i < 1 * raw.shape[0] / 3:
                    # n = n * pow(10, (p_s - p_n - poison_snr-poison_std) / 10.)
                    if poison_fold == 3 or poison_fold == 0: column = column + n
                elif i < 2 * raw.shape[0] / 3:
                    # n = n * pow(10, (p_s - p_n - poison_snr) / 10.)
                    if poison_fold == 2 or poison_fold == 0: column = column + n
                else:
                    # n = n * pow(10, (p_s - p_n - poison_snr+poison_std) / 10.)
                    if poison_fold == 1 or poison_fold == 0: column = column + n

            #print('Category {} | Segment: {:.2f} | SNR: {:.2f} '.format(category, i/ raw.shape[0], p_s - 10*np.log10(np.mean(n))))

            raw[i:i + length, :] = column

        length = int(length/2)


    # packing
    for i in range(0, raw.shape[0]-max(length,step), step):

        column = raw[i:i + length, :]

        # NONE
        if normalize == 0:
            #
            column = np.log10(column + np.finfo(float).eps)

        # AU
        if normalize == 1:
            column = np.log10(column + np.finfo(float).eps)
            column = column + np.random.uniform(-au / 10., au / 10.)

        # Per band gain normalization
        if normalize == 2:
            column = np.log10(column + np.finfo(float).eps)
            for k in range(0, raw.shape[1], 1):
                column[:, k] = raw[i:i + length, k] - np.mean(column)

        # MEANVAR
        if normalize == 3:
            column = np.log10(column + np.finfo(float).eps)
            column = column - np.mean(column)
            column = column / np.std(column)
            column = column + np.random.uniform(-persa_au / 10., persa_au / 10.)

        # PCEN
        if normalize == 4:
            #
            column = column + np.random.uniform(-pcen_au / 10., pcen_au / 10.)

        # PONS
        if normalize == 7:
            column = column / np.amax(column + np.finfo(float).eps)
            column = np.log10(column + pow(10, -pons_snr / 10.))
            # column = column + np.random.uniform(-pons_au / 10., pons_au / 10.)

        # PERSA
        if normalize == 8:
            column = np.log10(column + np.finfo(float).eps)
            column = column - np.mean(column)
            column = column + np.random.uniform(-persa_au / 10., persa_au / 10.)
            # column = np.rint(3 * column)
            # print(np.amax(column)-np.amin(column))

        # PERSA+
        if normalize == 9:

            if persa_auto:

                #snr = 10 * np.log10((np.percentile(column.flatten(), 25) + np.finfo(float).eps) / (np.percentile(column.flatten(), 5) + np.finfo(float).eps))
                r = 10 * np.log10(column[(column > np.percentile(column.flatten(), 5)) & (column < np.percentile(column.flatten(), 50))])
                snr = np.std(r) * 1

                persa_snr = 15
                if snr < 9: persa_snr = 9
                #if snr < 6: persa_snr = 6

                metric[int(i/step), 0] = snr
                metric[int(i/step), 1] = persa_snr
                #print(category, ' ',  metric[int(i / step), 2])

                # log_column = 10 * np.log10(column + np.finfo(float).eps)
                # flat = gmean(column, axis=1) / (np.mean(column, axis=1) + np.finfo(float).eps)
                # flat = np.mean(flat)
                # metric[int(i / step), 1] = flat

                # log_column = log_column - np.mean(log_column)
                # per_10 = np.percentile(np.mean(log_column, axis=0).flatten(), 5)
                # persa_snr = max(12, np.mean(log_column.flatten()) - np.percentile(np.mean(log_column, axis=1).flatten(), 10))

                # q = log_column < -12
                # np.mean(log_column[q])
                # hist, bins = np.histogram(log_column, bins=[-92, -18, -15, -12, -9, -6, -3], density=False)
                # metric[int(i / step), 0] = np.percentile(log_column.flatten(), 90)
                # metric[int(i / step), 1] = np.percentile(log_column.flatten(), 50)
                # metric[int(i / step), 2] = np.percentile(log_column.flatten(), 10)
                # metric[int(i / step), 3] = 0
                # metric[int(i / step), 4] = 0

                # x = np.random.normal(0.0, 0.1, 50000)
                # x = np.random.uniform(-1, 1, 50000)
                # mel_n = librosa.feature.melspectrogram(x, 22050, n_fft=512, hop_length=256, power=2.0, n_mels=56, fmin=100, fmax=8000)
                # mel_n = np.transpose(mel_n)
                # n = mel_n[:length, :]

            #x = np.random.normal(0.0, 0.1, 44100)
            #mel_n = np.transpose(librosa.feature.melspectrogram(x, x_rate, n_fft=512, hop_length=256, power=2.0, n_mels=56, fmin=100, fmax=8000))
            #n = mel_n[:length, :]
            #n_pos = np.random.randint(0, mel_n.shape[0]-length)
            #n = mel_n[n_pos:n_pos+length, :]
            #if i <
            #    n = mel_n[i:i + length, :]
            #else:
            #    print(i, mel_n.shape[0])
            n = mel_n[i%(mel_n.shape[0]-length):i%(mel_n.shape[0]-length)+length, :]

            # for j in range(0, column.shape[0]-20):
            #     if j == 0: hp_s = -90
            #     if 10 * np.log10(np.mean(column[j:j+20, :])) > hp_s: hp_s = 10 * np.log10(np.mean(column[j:j+20, :]))
            # p_s = hp_s-3
            p_s = 10 * np.log10(np.mean(column)+np.finfo(float).eps)
            p_n = 10 * np.log10(np.mean(n)+np.finfo(float).eps)
            n = n * pow(10, (p_s - p_n - persa_snr) / 10.)
            column = column + n
            #print('Signal PW: {:.2f} | Noise PW: {:.2f} | Noise PWC: {:.2f} '.format(p_s, p_n, 10 * np.log10(np.mean(n))))

            column = np.log10(column + np.finfo(float).eps)
            column = column - np.mean(column)
            column = column + np.random.uniform(-persa_au / 10., persa_au / 10.)
            # column = np.rint(3 * column)

        data[int(i/step), :, :] = column
        #data[2*int(i/step)+1, :, :] = column + np.random.uniform(-au / 10., au / 10.)

    data = data[:data.shape[0]-1, :, :]
    truth = truth[:truth.shape[0]-1, :]

    '''data = np.append(data, data, axis=0)
    truth = np.append(truth, truth, axis=0)

    ll = int(data.shape[0]/2)
    for i in range(0, ll):
        data[i+ll, :, :] = data[i, :, :] + np.random.uniform(-au / 10., au / 10.)
        truth[i+ll, :] = truth[i, :]'''


    # print info
    if category == 0:
        print('Loading data as TF textures')
        print('Normalization method is {}'.format(normalize))
        if poison_fold >= 0 and poison_type == 'eq':    print('Dataset is distored with -1dB/band for the first {} bands at fold ()'.format(eq_snr, poison_fold))
        if poison_fold >= 0 and poison_type == 'snr':   print('Dataset is distored with {}±{} dB SNR at fold {}'.format(poison_snr, poison_std, poison_fold))
        print('Texture length is {0:.2f}s, step {1:.2f}s and number of bands {2:.0f}'.format(length * fft_step / sample_rate, step / length, raw.shape[1]))
        # print("Class {} data has mea|std min|max {:.2f}|{:.2f} {:.2f}|{:.2f}".format(category, np.mean(data), np.std(data), np.amin(data), np.amax(data)))
        # print('M1|M2|M3 {:.2f}|{:.2f}|{:.2f}'.format(np.mean(metric[:, 0]), np.mean(metric[:, 1]), np.mean(metric[:, 2])))


    return data, truth

def load_spectrum_csv_ts(path, category):

    # timing configuration
    sample_rate = 22050
    fft_step = 512
    em_length = 12
    em_step = int(em_length * 0.5)
    tx_length = 9
    tx_step = int(tx_length * 0.5)

    # normalization configuration
    normalize = 9  # 0 None, 1 Patch, 2 Band, 3 Band Global, 4 PCEN, 9 PERSA+
    persa_snr = 9
    x, x_rate = librosa.load('D:\\PhD\\Datasets\\pink_noise.wav', sr=sample_rate, mono=True)
    mel_n = np.transpose(librosa.feature.melspectrogram(x, x_rate, n_fft=512, hop_length=256, power=2.0, n_mels=56, fmin=100, fmax=8000))

    # texture formation
    raw = np.genfromtxt(path, delimiter=',')
    textures = np.zeros((int(raw.shape[0]/tx_step), tx_length, raw.shape[1]))
    for i in range(0, raw.shape[0]-tx_length, tx_step):

        texture = raw[i:i + tx_length, :]

        if normalize == -1:
            texture = np.log10(texture + np.finfo(float).eps)
            texture = texture - np.mean(texture)

        textures[int(i/tx_step), :, :] = texture

    # embedding formation
    data = np.zeros((int(raw.shape[0]/tx_step/em_step), em_length, tx_length, raw.shape[1]))
    truth = np.full(int(raw.shape[0]/tx_step/em_step), category)
    truth = truth.reshape(truth.shape[0], 1)
    for i in range(0, int(raw.shape[0]/tx_step)-em_length, em_step):

        embedding = textures[i:i+em_length, :, :]

        if normalize == 0:
            #
            embedding = np.log10(embedding + np.finfo(float).eps)

        if normalize == 1:
            embedding = np.log10(embedding + np.finfo(float).eps)
            embedding = embedding - np.mean(embedding)

        if normalize == 9:
            n_pos = np.random.randint(0, mel_n.shape[0]-tx_length*em_length)
            n = mel_n[n_pos:n_pos+tx_length*em_length, :]

            p_s = 10 * np.log10(np.mean(embedding))
            p_n = 10 * np.log10(np.mean(n))
            n = n * pow(10, (p_s - p_n - persa_snr) / 10.)

            embedding = embedding + n.reshape(embedding.shape)
            embedding = np.log10(embedding + np.finfo(float).eps)
            embedding = embedding - np.mean(embedding)

        data[int(i/em_step), :, :, :] = embedding

    # print info
    if category == 0:
        print('Texture Length: {0:.2f}s'.format(tx_length*fft_step/sample_rate))
        print('Sequence Length: {0:.2f}s'.format(em_length*tx_step*fft_step/sample_rate))
    print("Class {} with mean|std min|max {:.2f}|{:.2f} {:.2f}|{:.2f}".format(category,np.mean(data),np.std(data),np.amin(data),np.amax(data)))

    return data, truth


# Load spectrograms from image
def load_img_ts(path, category):

    # timing configuration
    em_length = 10
    em_step = 10

    files = os.listdir(path)
    files = sorted(files)

    # texture formation
    raw = np.zeros((len(files), 30, 50))
    for i in range(0, len(files), 1):
        image = img.open(path+'\\'+files[i])
        image = image.resize((50, 30), img.ANTIALIAS)
        # plt.imshow(image)
        # plt.show()
        # print(path+'\\'+files[i])
        raw[i, :, :] = np.asarray(image)
        raw[i, :, :] = raw[i, :, :]/255

    # embedding formation
    data = np.zeros((int(raw.shape[0]/em_step), em_length, raw.shape[1], raw.shape[2]))
    truth = np.full(int(raw.shape[0]/em_step), category)
    for i in range(0, raw.shape[0]-em_length-1, em_step):
        data[int(i/em_step), :, :, :] = raw[i:i+em_length, :, :]
        # print("{} -> {}".format(int(i/em_step), np.mean(data[int(i/em_step), :, :, :])))
        # print(truth[i])

    print('Category {} with {} items'.format(category, data.shape[0]))

    return data, truth


# Utility
def preemphasis(y, coef=0.97):
    return scipy.signal.lfilter([1.0, -coef], [1.0], y)

def remove_silent_frames(x, y, fs, dyn_range=30, framelen=4096, hop=1024):
    """
    Remove silent frames of x and y based on x
    A frame is excluded if its energy is lower than max(energy) - dyn_range
    The frame exclusion is based solely on x, the clean speech signal
    # Arguments :
        x : array, original speech wav file
        y : array, denoised speech wav file
        dyn_range : Energy range to determine which frame is silent. Speech dynamic range is around 40 dB
        framelen : Window size for energy evaluation
        hop : Hop size for energy evaluation
    # Returns :
        x without the silent frames
        y without the silent frames (aligned to x)
    """
    EPS = np.finfo("float").eps
    # Compute Mask
    import scipy as sp
    w = sp.hanning(framelen + 2)[1:-1]

    x_frames = np.array(
        [w * x[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    y_frames = np.array(
        [w * y[i:i + framelen] for i in range(0, len(x) - framelen, hop)])

    # Compute energies in dB
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)

    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0

    # Remove silent frames by masking
    x_frames = x_frames[mask]
    y_frames = y_frames[mask]

    # init zero arrays to hold x, y with silent frames removed
    x_sil = np.zeros(x_frames.shape[0] * hop + framelen) # np.zeros((mask.shape[0] - 1) * hop + framelen)
    y_sil = np.zeros(x_frames.shape[0] * hop + framelen) # np.zeros((mask.shape[0] - 1) * hop + framelen)
    for i in range(x_frames.shape[0]):
        x_sil[range(i * hop, i * hop + framelen)] += x_frames[i, :]
        y_sil[range(i * hop, i * hop + framelen)] += y_frames[i, :]

    return x_sil / 2., y_sil / 2.


# 1D FSDD
def get_audio_dataset_1d(folder):

    size = 4000
    agm = 1
    samples = np.zeros((size, agm * 2000))
    labels = np.zeros(0)
    speakers = np.zeros(0)

    avg_size = 0
    items = 0

    for i, f in enumerate(os.listdir(folder)):
        if f.endswith('.wav'):

            sample, sr = librosa.load(folder +'/' + f, sr=None , mono=True)

            avg_size = sample.shape[0] + avg_size
            sample = sample / np.sqrt(np.mean(np.square(sample)))
            # sign = np.sign(sample)
            # sample = np.log10(np.abs(sample) + 1)
            # sample = np.multiply(sample, sign)

            if sample.shape[0] > size:
                #
                sample = sample[:size]
            else:
                pad = size - sample.shape[0]
                left = int(pad / 2)
                if pad % 2 == 1:
                    right = left + 1
                else:
                    right = left
                sample = np.pad(sample, (left, right), mode='constant')
            sample = sample.reshape((sample.shape[0], 1))

            label, speaker, trial, = f.split('_')[0], f.split('_')[1], f.split('_')[2][0]

            for k in range(0, agm):
                if k > 0:
                    pad = np.random.randint(0, 100)
                    sample = sample[pad:]
                    sample = np.pad(sample,(0, pad), mode='constant')
                    #sample = sample * np.random.uniform(0.5, 1.5)
                samples[:, items] = sample[:, 0]
                labels = np.append(labels, label)
                speakers = np.append(speakers, speaker)
                items = 1 + items

    return samples, labels, speakers


# 2D FSDD ±
def get_audio_dataset_2d(folder, test_speaker, norm):

    size = 36
    data = np.zeros((size,size,0))
    labels = np.zeros(0)
    speakers = np.zeros(0)

    global mel_nw
    sp, sr = librosa.load('D:\\PhD\\Datasets\\LVLib-SMO-v1\\PCM\\music.wav', sr=8000, mono=True)
    mel_nw = librosa.feature.melspectrogram(sp, sr=8000, n_fft=256, hop_length=128, power=2.0, n_mels = size, fmin=80, fmax=4000)

    for i, f in enumerate(os.listdir(folder)):
        if f.endswith('.wav'):
            # print(f)
            sample = wav2melspectrogram(folder +'/' + f, size, test_speaker, norm=norm)
            label, speaker, trial, = f.split('_')[0], f.split('_')[1], f.split('_')[2][0]
            for k in range(0, 3):
                r = np.random.randint(-5, 5)
                if r < 0:   sample = np.pad(sample[:, :r], pad_width=((0, 0), (-r, 0)), mode='constant')
                if r > 0:   sample = np.pad(sample[:, r:], pad_width=((0, 0), (0, r)), mode='constant')
                data = np.append(data, sample.reshape((sample.shape[0], sample.shape[1], 1)), axis=2)
                labels = np.append(labels, int(label))
                speakers = np.append(speakers, speaker)

    return data, labels, speakers

def wav2melspectrogram(file_path, max_pad_len, test_speaker, poison_snr=30, norm=0): # 0 LOG, 1 LOG-AU, 4 PCEN, 7 PONS, 8 PERSA, 9 PERSA+

    pons_snr = 30
    persa_snr = 9

    # print info
    if file_path.find('0_jackson_0') > 0: print('\nLoading data with normalization method {}'.format(norm))

    # form sample
    sp, sr = librosa.load(file_path, mono=True, sr=8000)
    mel_sp = librosa.feature.melspectrogram(sp, sr, n_fft=256, hop_length=128, power=2.0, n_mels=max_pad_len, fmin=80, fmax=4000)

    # pad sample
    if max_pad_len < mel_sp.shape[1]:
        #
        mel_sp = mel_sp[:, :max_pad_len]
    else:
        pad = max_pad_len - mel_sp.shape[1]
        left = int(pad / 2)
        if pad % 2 == 1:    right = left + 1
        else:               right = left
        mel_sp = np.pad(mel_sp, pad_width=((0, 0), (left, right)), mode='constant')

    # poison sample
    if file_path.find(test_speaker) > 0 and poison_snr < 18:
        if file_path.find('0_') > 0 and file_path.find('_0.wav') > 0: print('Poisoning', test_speaker)
        global mel_nw
        n_pos = np.random.randint(0, mel_nw.shape[1] - mel_sp.shape[1])
        p_s = 10 * np.log10(np.mean(mel_sp))
        # for i in range(0, mel_sp.shape[1]-10):
        #     if i == 0: pw = -90
        #     if 10 * np.log10(np.mean(mel_sp[:, i:i+10])) > pw: pw = 10 * np.log10(np.mean(mel_sp[:, i:i+10]))
        # print('PW Diff: {:.2f}'.format(pw - p_s))
        p_n = 10 * np.log10(np.mean(mel_nw[:, n_pos:n_pos + mel_sp.shape[1]]))
        mel_sp = mel_sp + mel_nw[:, n_pos:n_pos + mel_sp.shape[1]] * pow(10, (p_s - p_n - np.random.normal(poison_snr, 3)) / 10.)
        mel_sp = mel_sp * pow(10, (np.random.normal(0, 10)) / 10.)

    #  LOG
    if norm == 0:
        #
        mel_sp = np.log10(mel_sp + np.finfo(float).eps)

    # AU
    if norm == 1:
        mel_sp = np.log10(mel_sp + np.finfo(float).eps)
        mel_sp = mel_sp + np.random.uniform(-30 / 10., 30 / 10.)

    # PCEN
    if norm == 4:
        mel_sp = np.sqrt(mel_sp)
        mel_sp = librosa.pcen(mel_sp * (2**31), sr=sr, hop_length=128)

    #  PONS
    if norm == 7:
        mel_sp = mel_sp / np.amax(mel_sp)
        mel_sp = np.log10(mel_sp + pow(10, -pons_snr / 10.))

    # PERSA
    if norm == 8:
        mel_sp = np.log10(mel_sp + np.finfo(float).eps)
        mel_sp = mel_sp - np.mean(mel_sp)
        mel_sp = mel_sp + np.random.uniform(-3 / 10., 3 / 10.)

    # PERSA+
    if norm == 9:
        x = np.random.normal(0.0, 0.1, 24000)
        mel_n = librosa.feature.melspectrogram(x, sr, n_fft=256, hop_length=128, power=2.0, n_mels=max_pad_len, fmin=80, fmax=4000)
        mel_n = mel_n[:, :mel_sp.shape[1]]
        p_s = 10 * np.log10(np.mean(mel_sp))
        p_n = 10 * np.log10(np.mean(mel_n))
        mel_n = mel_n * pow(10, (p_s - p_n - persa_snr) / 10.)
        # p_n_c = 10 * np.log10(np.mean(mel_n))
        # print('Signal PW: {:.2f} | Noise PW: {:.2f} | Noise PWC: {:.2f} '.format(p_s, p_n, p_n_c))
        mel_sp = mel_sp + mel_n
        mel_sp = np.log10(mel_sp + np.finfo(float).eps)
        mel_sp = mel_sp - np.mean(mel_sp)
        mel_sp = mel_sp + np.random.uniform(-3 / 10., 3 / 10.)

    # mel_n = np.transpose(mel_n)
    return mel_sp