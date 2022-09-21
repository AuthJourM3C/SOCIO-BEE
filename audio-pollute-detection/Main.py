# Imports
import os, warnings, time, math, librosa
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, GenericUnivariateSelect, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
import HandleInput as hi
import tensorflow as tf
# import TorchCNN2D as tcnn2d
import DeepANN as ann, DeepANNLSTM as annlstm
import DeepCNN1D as cnn1d, DeepCNN2D as cnn2d
import DeepCNN1DLSTM as cnn1dlstm, DeepCNN2DLSTM as cnn2dlstm


# Utility
def configure_keras():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # from keras import backend as K
    # config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(),config=config)
    # K.set_session(sess)
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    print('Detected GPUs:', tf.test.gpu_device_name())
    # print(tf.python.client.device_lib.list_local_devices())
def get_array_part(array, start, stop):
    length = array.shape[0]
    return array[int(start*length):int(stop*length)]
def transform(path, mode):
    files = os.listdir(path)
    for file in files:
        if '.wav' in file:
            if mode == 'mel':
                hi.save_mel_spectrogram(path, file[:-4], 22050, 512, 256, 56, 100, 8000, False)
            if mode == 'pcen':
                hi.save_mel_spectrogram(path, file[:-4], 22050, 512, 256, 56, 100, 8000, True)
            if mode == 'stft':
                hi.save_stft_spectrogram(path, file[:-4], 22050, 512, 256)
            if mode == 'stfv':
                hi.save_stfv(path, file[:-4], 22050, 512, 256)


# Initialization
configure_keras()

# FSDD
def cnn_1d_fsdd(test_speaker):
    data, labels, speakers = hi.get_audio_dataset_1d('D:\\PhD\Datasets\\FSDD\\recordings\\')
    labels = np.reshape(labels,(labels.shape[0], 1))
    speakers = np.reshape(speakers,(speakers.shape[0], 1))

    x_train = np.zeros((0, data.shape[0]))
    x_test = np.zeros((0, data.shape[0]))
    y_train = np.zeros((0, 1))
    y_test = np.zeros((0, 1))

    print(data.shape, labels.shape, speakers.shape)

    for i in range(0, labels.shape[0]):
        #print(speakers[i])

        sample = data[:, i]
        sample = sample.reshape((1, sample.shape[0]))

        if(test_speaker in speakers[i]):
            x_test = np.append(x_test, sample, axis=0)
            y_test = np.append(y_test, labels[i, 0])
        else:
            x_train = np.append(x_train, sample, axis=0)
            y_train = np.append(y_train, labels[i, 0])

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    score = cnn1d.train(x_train, y_train, x_test, y_test, 200, batch_size=8)
    print('Test accuracy: {:.1f}'.format(100*score))

    return score
def cnn_2d_fsdd(test_speaker, norm):

    spectrograms, labels, speakers = hi.get_audio_dataset_2d('D:\\PhD\\Datasets\\FSDD\\recordings\\', test_speaker, norm)
    labels = np.reshape(labels,(labels.shape[0], 1))
    speakers = np.reshape(speakers,(speakers.shape[0], 1))

    x_train = np.zeros((0, spectrograms.shape[0], spectrograms.shape[1]))
    x_test = np.zeros((0, spectrograms.shape[0], spectrograms.shape[1]))
    y_train = np.zeros((0, 1))
    y_test = np.zeros((0, 1))

    print(spectrograms.shape, labels.shape, speakers.shape)

    for i in range(0, labels.shape[0]):
        #print(speakers[i])

        spec = spectrograms[:, :, i]
        spec = spec.reshape((1, spec.shape[0], spec.shape[1]))

        if test_speaker in speakers[i]:
            x_test = np.append(x_test, spec, axis=0)
            y_test = np.append(y_test, labels[i, 0])
        else:
            x_train = np.append(x_train, spec, axis=0)
            y_train = np.append(y_train, labels[i, 0])

    score = cnn2d.train(x_train, y_train, x_test, y_test, 200)
    # score = tcnn2d.train(x_train, y_train, x_test, y_test, 100)

    print('Test accuracy: {:.1f}'.format(100*score))

    return score
for k in []:                                        # front-ends
    for j in ['jackson', 'yweweler', 'theo', 'nicolas']:
        for i in range(1, 4):
            if i==1: acc = np.zeros((0))
            #acc[i-1] = cnn_1d_fsdd('nicolas')  # jackson yweweler theo nicolas  ±
            acc = np.append(acc, cnn_2d_fsdd(j, k))
            if i==3: print('Accuracy: {:.1f} ±{:.1f} for front-end {}'.format(100 * np.mean(acc), 100 * np.std(acc), k))

# LVLib
def ann_smo_lvlib(mode, dataset, epochs, fold, folds):

    # folds
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load data
    if mode == 'rnn':
        # load dataset
        if dataset.find('v') >= 0:
            if dataset.find('v3') >= 0:     path = 'D:\\PhD\\Datasets\\LVLib-SMO-v3\\STFV\\'
            elif dataset.find('v4') >= 0:   path = 'D:\\PhD\\Datasets\\LVLib-SMO-v4\\STFV\\'
            else:                           return 0
            print('Data from:', path)
            for s in ["mfccs"]:
                x_0, y_0 = hi.load_features_csv(path + 'music.wav.norm-' + s + '.csv', 0)
                x_1, y_1 = hi.load_features_csv(path + 'others.wav.norm-' + s + '.csv', 1)
                x_2, y_2 = hi.load_features_csv(path + 'speech.wav.norm-' + s + '.csv', 2)
            for s in ["p-sha", "p-spr", "s-dec", "s-fla", "s-flu", "s-rol", "s-sha", "s-slo", "s-var", "t-zcr"]:
                kx_0, _ = hi.load_features_csv(path + 'music.wav.norm-' + s + '.csv', 0)
                kx_1, _ = hi.load_features_csv(path + 'others.wav.norm-' + s + '.csv', 1)
                kx_2, _ = hi.load_features_csv(path + 'speech.wav.norm-' + s + '.csv', 2)
                x_0 = np.concatenate((x_0, kx_0), axis=1)
                x_1 = np.concatenate((x_1, kx_1), axis=1)
                x_2 = np.concatenate((x_2, kx_2), axis=1)
            x_train = np.concatenate((get_array_part(x_0, 0.0, s1),
                                      get_array_part(x_1, 0.0, s1),
                                      get_array_part(x_2, 0.0, s1),
                                      get_array_part(x_0, s2, 1.0),
                                      get_array_part(x_1, s2, 1.0),
                                      get_array_part(x_2, s2, 1.0)), axis=0)
            y_train = np.concatenate((get_array_part(y_0, 0.0, s1),
                                      get_array_part(y_1, 0.0, s1),
                                      get_array_part(y_2, 0.0, s1),
                                      get_array_part(y_0, s2, 1.0),
                                      get_array_part(y_1, s2, 1.0),
                                      get_array_part(y_2, s2, 1.0)), axis=0)
            x_test = np.concatenate((get_array_part(x_0, s1, s2),
                                     get_array_part(x_1, s1, s2),
                                     get_array_part(x_2, s1, s2),), axis=0)
            y_test = np.concatenate((get_array_part(y_0, s1, s2),
                                     get_array_part(y_1, s1, s2),
                                     get_array_part(y_2, s1, s2)), axis=0)
        # load additional dataset
        if dataset.find('v3-v4') >= 0:
            path = 'D:\\PhD\\Datasets\\LVLib-SMO-v4\\STFV\\'
            print('Data from:', path)
            for s in ["mfccs"]:
                x_0, y_0 = hi.load_features_csv(path + 'music.wav.norm-' + s + '.csv', 0)
                x_1, y_1 = hi.load_features_csv(path + 'others.wav.norm-' + s + '.csv', 1)
                x_2, y_2 = hi.load_features_csv(path + 'speech.wav.norm-' + s + '.csv', 2)
            for s in ["p-sha", "p-spr", "s-dec", "s-fla", "s-flu", "s-rol", "s-sha", "s-slo", "s-var", "t-zcr"]:
                kx_0, _ = hi.load_features_csv(path + 'music.wav.norm-' + s + '.csv', 0)
                kx_1, _ = hi.load_features_csv(path + 'others.wav.norm-' + s + '.csv', 1)
                kx_2, _ = hi.load_features_csv(path + 'speech.wav.norm-' + s + '.csv', 2)
                x_0 = np.concatenate((x_0, kx_0), axis=1)
                x_1 = np.concatenate((x_1, kx_1), axis=1)
                x_2 = np.concatenate((x_2, kx_2), axis=1)
            x_test = np.concatenate((get_array_part(x_0, s1, s2),
                                     get_array_part(x_1, s1, s2),
                                     get_array_part(x_2, s1, s2),), axis=0)
            y_test = np.concatenate((get_array_part(y_0, s1, s2),
                                     get_array_part(y_1, s1, s2),
                                     get_array_part(y_2, s1, s2)), axis=0)

        # remove unwanted features
        r = [0]
        print('Number of initial features: {}'.format(x_train.shape[1]))
        print('Mode: {} excluding features: {}'.format(mode, r))
        x_train = np.delete(x_train, r, axis=1)
        x_test = np.delete(x_test, r, axis=1)
        print('Number of intermediate features: {}'.format(x_train.shape[1]))

        # standardize
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        # reform to sequential data
        ts_length = 128
        ts_step = 64
        ts_data = np.zeros((int(x_train.shape[0] / ts_step), ts_length, x_train.shape[1]))
        ts_truth = np.zeros(int(y_train.shape[0] / ts_step))
        for i in range(0, x_train.shape[0] - ts_length, ts_step):
            ts_data[int(i / ts_step), :, :] = x_train[i:i + ts_length, :]
            ts_truth[int(i / ts_step)] = y_train[i]
        x_train = ts_data
        y_train = ts_truth
        ts_data = np.zeros((int(x_test.shape[0] / ts_step), ts_length, x_test.shape[1]))
        ts_truth = np.zeros(int(y_test.shape[0] / ts_step))
        for i in range(0, x_test.shape[0] - ts_length, ts_step):
            ts_data[int(i / ts_step), :, :] = x_test[i:i + ts_length, :]
            ts_truth[int(i / ts_step)] = y_test[i]
        x_test = ts_data
        y_test = ts_truth

        # fit or predict
        score = annlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    else:
        # load dataset
        if dataset.find('v3') >=0 or dataset.find('v4')>= 0:
            if dataset.find('v3') >=0:      path = 'D:\\PhD\\Datasets\\LVLib-SMO-v3\\ADi\\'
            elif dataset.find('v4') >= 0:   path = 'D:\\PhD\\Datasets\\LVLib-SMO-v4\\ADi\\'
            print('Data from:', path)
            x_0, y_0 = hi.load_features_csv(path + 'music.csv', 0)
            x_1, y_1 = hi.load_features_csv(path + 'others.csv', 1)
            x_2, y_2 = hi.load_features_csv(path + 'speech.csv', 2)
            x_train = np.concatenate((get_array_part(x_0, 0.0, s1),
                                      get_array_part(x_1, 0.0, s1),
                                      get_array_part(x_2, 0.0, s1),
                                      get_array_part(x_0, s2, 1.0),
                                      get_array_part(x_1, s2, 1.0),
                                      get_array_part(x_2, s2, 1.0)), axis=0)
            y_train = np.concatenate((get_array_part(y_0, 0.0, s1),
                                      get_array_part(y_1, 0.0, s1),
                                      get_array_part(y_2, 0.0, s1),
                                      get_array_part(y_0, s2, 1.0),
                                      get_array_part(y_1, s2, 1.0),
                                      get_array_part(y_2, s2, 1.0)), axis=0)
            x_test = np.concatenate((get_array_part(x_0, s1, s2),
                                     get_array_part(x_1, s1, s2),
                                     get_array_part(x_2, s1, s2),), axis=0)
            y_test = np.concatenate((get_array_part(y_0, s1, s2),
                                     get_array_part(y_1, s1, s2),
                                     get_array_part(y_2, s1, s2)), axis=0)
        # load additional dataset
        if dataset.find('v3-v4') >= 0:
            path = 'D:\\PhD\\Datasets\\LVLib-SMO-v4\\ADi\\'
            print('Data from:', path)
            x_0, y_0 = hi.load_features_csv(path + 'music.csv', 0)
            x_1, y_1 = hi.load_features_csv(path + 'others.csv', 1)
            x_2, y_2 = hi.load_features_csv(path + 'speech.csv', 2)
            x_test = np.concatenate((get_array_part(x_0, s1, s2),
                                     get_array_part(x_1, s1, s2),
                                     get_array_part(x_2, s1, s2),), axis=0)
            y_test = np.concatenate((get_array_part(y_0, s1, s2),
                                     get_array_part(y_1, s1, s2),
                                     get_array_part(y_2, s1, s2)), axis=0)

        # remove unwanted features
        r = []
        r.extend([204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215])
        r.extend([288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299])
        if mode == 'sti':
            for i in range(x_train.shape[1]):
                if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                    r.append(i)
            r.append(0)
        elif mode == 'adi':
            for i in range(x_train.shape[1]):
                if i % 12 == 0 or i % 12 == 1 or i % 12 == 2 or i % 12 == 3 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                    r.append(i)
            r.append(7)
        elif mode == 'sti-eti':
            for i in range(x_train.shape[1]):
                if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7:
                    r.append(i)
            r.append(0)
        elif mode == 'adi-eti':
            for i in range(x_train.shape[1]):
                if i % 12 == 0 or i % 12 == 1 or i % 12 == 4 or i % 12 == 5:
                    r.append(i)
            r.append(7)
        else:
            #
            return 0
        print('Number of initial features: {}'.format(x_train.shape[1]))
        print('Mode: {} excluding features: {}'.format(mode, r))
        x_train = np.delete(x_train, r, axis=1)
        x_test = np.delete(x_test, r, axis=1)
        print('Number of intermediate features: {}'.format(x_train.shape[1]))

        # standardize
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        #  feature selection
        # selector = SelectFromModel(DecisionTreeClassifier(), threshold=-np.inf, max_features=min(40, x_train_lvlib.shape[1]))
        # selector = GenericUnivariateSelect(f_classif, mode='fdr')  # fpr, fdr, fwe, ('k_best', param=100) mutual_info_classif f_classif
        # selector.fit(x_train_lvlib, y_train_lvlib)
        # x_train_lvlib = selector.transform(x_train_lvlib)
        # x_test_lvlib = selector.transform(x_test_lvlib)
        # print('Number of selected features: {}'.format(x_train_lvlib.shape[1]))

        # fit or predict
        score = ann.train(x_train, y_train, x_test, y_test, 3, epochs)
        # model = LogisticRegression(random_state=0, multi_class='auto', solver='lbfgs').fit(x_train_lvlib, y_train_lvlib)
        # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (64, 32), random_state = 1)

    print('Fold {}, mode {}, accuracy: {}'.format(fold, mode, round(100 * score, 1)))
    return score
def cnn_1d_smo_lvlib(mode, dataset, few_shot, normalize, lstm, epochs, fold, folds):

    if normalize == 4 or normalize == 7:
        print('\nThis front-end is not supported in 1D architecture, skipping...')
        return 0

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load primary dataset
    if dataset.find('v') == 0:

        # load data
        path = 'D:\\PhD\\Datasets\\LVLib-SMO-' + dataset[:2] + '\\PCM\\'
        print('\nData from:', path)
        if lstm:
            x_music, y_music = hi.load_audio_ts(path + 'music.wav', 0)
            x_speech, y_speech = hi.load_audio_ts(path + 'speech.wav', 1)
            x_other, y_other = hi.load_audio_ts(path + 'others.wav', 2)
        else:
            x_music, y_music = hi.load_audio(path + 'music.wav', 0, normalize)
            x_speech, y_speech = hi.load_audio(path + 'speech.wav', 1, normalize)
            x_other, y_other = hi.load_audio(path + 'others.wav', 2, normalize)

        # make folds
        x_train = np.concatenate((get_array_part(x_music, 0, s1),
                                  get_array_part(x_speech, 0, s1),
                                  get_array_part(x_other, 0, s1),
                                  get_array_part(x_music, s2, 1),
                                  get_array_part(x_speech, s2, 1),
                                  get_array_part(x_other, s2, 1)), axis=0)
        y_train = np.concatenate((get_array_part(y_music, 0, s1),
                                  get_array_part(y_speech, 0, s1),
                                  get_array_part(y_other, 0, s1),
                                  get_array_part(y_music, s2, 1),
                                  get_array_part(y_speech, s2, 1),
                                  get_array_part(y_other, s2, 1)), axis=0)
        x_test = np.concatenate((get_array_part(x_music, s1, s2),
                                 get_array_part(x_speech, s1, s2),
                                 get_array_part(x_other, s1, s2)), axis=0)
        y_test = np.concatenate((get_array_part(y_music, s1, s2),
                                 get_array_part(y_speech, s1, s2),
                                 get_array_part(y_other, s1, s2)), axis=0)
    # load secondary dataset
    if dataset.find('-') > 0 or dataset.find('+') > 0:

        # load data
        path = 'D:\\PhD\\Datasets\\LVLib-SMO-' + dataset[-2:] + '\\PCM\\'
        print('\nExtra data from:', path)
        if lstm:
            x_music, y_music = hi.load_audio_ts(path + 'music.wav', 0)
            x_speech, y_speech = hi.load_audio_ts(path + 'speech.wav', 1)
            x_other, y_other = hi.load_audio_ts(path + 'others.wav', 2)
        else:
            x_music, y_music = hi.load_audio(path + 'music.wav', 0, normalize)
            x_speech, y_speech = hi.load_audio(path + 'speech.wav', 1, normalize)
            x_other, y_other = hi.load_audio(path + 'others.wav', 2, normalize)

        #  make folds
        if dataset.find('+') > 0:
            print('\nTraining and evaluating on LVLib', dataset)
            x_train = np.concatenate((get_array_part(x_music, 0, s1),
                                      get_array_part(x_speech, 0, s1),
                                      get_array_part(x_other, 0, s1),
                                      get_array_part(x_music, s2, 1),
                                      get_array_part(x_speech, s2, 1),
                                      get_array_part(x_other, s2, 1),
                                      x_train), axis=0)
            y_train = np.concatenate((get_array_part(y_music, 0, s1),
                                      get_array_part(y_speech, 0, s1),
                                      get_array_part(y_other, 0, s1),
                                      get_array_part(y_music, s2, 1),
                                      get_array_part(y_speech, s2, 1),
                                      get_array_part(y_other, s2, 1),
                                      y_train), axis=0)
            x_test = np.concatenate((get_array_part(x_music, s1, s2),
                                     get_array_part(x_speech, s1, s2),
                                     get_array_part(x_other, s1, s2),
                                     x_test), axis=0)
            y_test = np.concatenate((get_array_part(y_music, s1, s2),
                                     get_array_part(y_speech, s1, s2),
                                     get_array_part(y_other, s1, s2),
                                     y_test), axis=0)
        if dataset.find('-') > 0:
            print('\nTraining on LVLib-SMO-{} and evaluating on LVLib-SMO-{}'.format(dataset[:2], dataset[-2:]))
            x_test = np.concatenate((get_array_part(x_music, s1, s2),
                                     get_array_part(x_speech, s1, s2),
                                     get_array_part(x_other, s1, s2)), axis=0)
            y_test = np.concatenate((get_array_part(y_music, s1, s2),
                                     get_array_part(y_speech, s1, s2),
                                     get_array_part(y_other, s1, s2)), axis=0)

    # global normalization
    '''rms = math.sqrt(np.square(x_train_lvlib).mean())
    x_train_lvlib = x_train_lvlib[:, :] / rms
    x_test_lvlib = x_test_lvlib[:, :] / rms'''

    # train
    if lstm:
        #
        score = cnn1dlstm.train(x_train, y_train, x_test, y_test, epochs)
    else:
        if mode == 'train':
            score = cnn1d.train(x_train, y_train, x_test, y_test, epochs, few_shot)
        elif mode == 'retrain':
            score = cnn1d.retrain(x_train, y_train, x_test, y_test, epochs)
        else:
            score = -1

    # print scores
    print('Fold {} with accuracy: {:.1f}'.format(fold, 100 * score))
    return score
def cnn_2d_smo_lvlib(mode, dataset, few_shot, normalize, lstm, epochs, fold, folds):

    """
    :param str mode: train, retrain, evaluate
    :param str dataset: v1, v2, v3, v4, v3-v4, v3+v4
    :param float few_shot: float (0, 1]
    :param int normalize: 0, 1, 4, 7, 8, 9
    :param bool lstm: if lstm architecture will be used
    :param int epochs: maximum number of epochs
    :param int fold: current fold
    :param int folds: total number of folds
    :return: score
    :rtype: float
    :raises ValueError: if the message_body exceeds 160 characters
    """

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print('\nCNN2D SMO')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load primary dataset
    if dataset.find('v') == 0:
        path = 'D:\\PhD\\Datasets\\LVLib-SMO-'+ dataset[:2] +'\\MEL\\'
        print('\nData from:', path)

        # load data
        if lstm:
            x_music, y_music = hi.load_spectrum_csv_ts(path + 'music.csv', 0)
            x_speech, y_speech = hi.load_spectrum_csv_ts(path + 'speech.csv', 1)
            x_other, y_other = hi.load_spectrum_csv_ts(path + 'others.csv', 2)
        else:
            poison = 'none'
            if poison == 'none':
                x_music, y_music = hi.load_spectrum_csv(path + 'music.csv', 0, normalize, -1)
                x_speech, y_speech = hi.load_spectrum_csv(path + 'speech.csv', 1, normalize,  -1)
                x_other, y_other = hi.load_spectrum_csv(path + 'others.csv', 2, normalize,  -1)
            if poison == 'test':
                x_music, y_music = hi.load_spectrum_csv(path + 'music.csv', 0, normalize, fold)
                x_speech, y_speech = hi.load_spectrum_csv(path + 'speech.csv', 1, normalize, fold)
                x_other, y_other = hi.load_spectrum_csv(path + 'others.csv', 2, normalize, fold)
            if poison == 'full':
                x_music, y_music = hi.load_spectrum_csv(path + 'music.csv', 0, normalize, 0)
                x_speech, y_speech = hi.load_spectrum_csv(path + 'speech.csv', 1, normalize, 0)
                x_other, y_other = hi.load_spectrum_csv(path + 'others.csv', 2, normalize, 0)

        #  make folds
        x_train = np.concatenate((get_array_part(x_music, 0, s1),
                                  get_array_part(x_speech, 0, s1),
                                  get_array_part(x_other, 0, s1),
                                  get_array_part(x_music, s2, 1),
                                  get_array_part(x_speech, s2, 1),
                                  get_array_part(x_other, s2, 1)), axis=0)
        y_train = np.concatenate((get_array_part(y_music, 0, s1),
                                  get_array_part(y_speech, 0, s1),
                                  get_array_part(y_other, 0, s1),
                                  get_array_part(y_music, s2, 1),
                                  get_array_part(y_speech, s2, 1),
                                  get_array_part(y_other, s2, 1)), axis=0)
        x_test = np.concatenate((get_array_part(x_music, s1, s2),
                                 get_array_part(x_speech, s1, s2),
                                 get_array_part(x_other, s1, s2)), axis=0)
        y_test = np.concatenate((get_array_part(y_music, s1, s2),
                                 get_array_part(y_speech, s1, s2),
                                 get_array_part(y_other, s1, s2)), axis=0)
    # load secondary dataset
    if dataset.find('-') > 0 or dataset.find('+') > 0:
        path = 'D:\\PhD\\Datasets\\LVLib-SMO-'+ dataset[-2:] +'\\MEL\\'
        print('\nExtra data from:', path)

        # load data
        if lstm:
            x_music, y_music = hi.load_spectrum_csv_ts(path + 'music.csv', 0)
            x_speech, y_speech = hi.load_spectrum_csv_ts(path + 'speech.csv', 1)
            x_other, y_other = hi.load_spectrum_csv_ts(path + 'others.csv', 2)
        else:
            x_music, y_music = hi.load_spectrum_csv(path + 'music.csv', 0, normalize, -1)
            x_speech, y_speech = hi.load_spectrum_csv(path + 'speech.csv', 1, normalize, -1)
            x_other, y_other = hi.load_spectrum_csv(path + 'others.csv', 2, normalize, -1)

        #  make folds
        if dataset.find('+') > 0:
            print('Training and evaluating on LVLib', dataset)
            x_train = np.concatenate((get_array_part(x_music, 0, s1),
                                      get_array_part(x_speech, 0, s1),
                                      get_array_part(x_other, 0, s1),
                                      get_array_part(x_music, s2, 1),
                                      get_array_part(x_speech, s2, 1),
                                      get_array_part(x_other, s2, 1),
                                      x_train), axis=0)
            y_train = np.concatenate((get_array_part(y_music, 0, s1),
                                      get_array_part(y_speech, 0, s1),
                                      get_array_part(y_other, 0, s1),
                                      get_array_part(y_music, s2, 1),
                                      get_array_part(y_speech, s2, 1),
                                      get_array_part(y_other, s2, 1),
                                      y_train), axis=0)
            x_test = np.concatenate((get_array_part(x_music, s1, s2),
                                     get_array_part(x_speech, s1, s2),
                                     get_array_part(x_other, s1, s2),
                                     x_test), axis=0)
            y_test = np.concatenate((get_array_part(y_music, s1, s2),
                                     get_array_part(y_speech, s1, s2),
                                     get_array_part(y_other, s1, s2),
                                     y_test), axis=0)
        if dataset.find('-') > 0:
            print('\nTraining on LVLib-SMO-{} and evaluating on LVLib-SMO-{}'.format(dataset[:2], dataset[-2:]))
            x_test = np.concatenate((get_array_part(x_music, s1, s2),
                                     get_array_part(x_speech, s1, s2),
                                     get_array_part(x_other, s1, s2)), axis=0)
            y_test = np.concatenate((get_array_part(y_music, s1, s2),
                                     get_array_part(y_speech, s1, s2),
                                     get_array_part(y_other, s1, s2)), axis=0)

    # x_train_lvlib, y_train_lvlib, x_test_lvlib, y_test
    # np.save('Temp\\lvlib_' + dataset + '_x_train', x_train_lvlib)
    # np.save('Temp\\lvlib_' + dataset + '_y_train', y_train_lvlib)
    # np.save('Temp\\lvlib_' + dataset + '_x_test', x_test_lvlib)
    # np.save('Temp\\lvlib_' + dataset + '_y_test', y_test)
    # return

    # train
    if lstm:
        #
        score = cnn2dlstm.train(x_train, y_train, x_test, y_test, epochs)
    else:
        if mode == 'train':
            score = cnn2d.train(x_train, y_train, x_test, y_test, epochs, few_shot)
            # score = tcnn2d.train(x_train_lvlib, y_train_lvlib, x_test_lvlib, y_test, epochs)
        elif mode == 'retrain':
            score = cnn2d.retrain(x_train, y_train, x_test, y_test, epochs)
        elif mode == 'evaluate':
            score = cnn2d.evaluate(x_test, y_test)
        else:
            score = -1

    print('Fold {} with accuracy: {:.1f}%'.format(fold, 100*score))
    return score
# transform('D:\PhD\Datasets\LVLib-SMO-v9\PCM\\', 'mel')  # mel, pcen, stfv
for k in [0,1,4,7,8,9]:                                                                         # normalize: 0 LOG, 1 LOG-AU, 4 PCEN, 7 LOG-T, 8 PERSA, 9 PERSA+
    for j in [1.00]:                                                                            # dataset size for few shot learning
        for i in range(4, 4):                                                                   # folds
            if 'acc' not in locals(): acc = np.zeros((3))
            # acc[i-1] = ann_smo_lvlib('adi-eti', 'v3-v4', 200, i, 3)
            acc[i-1] = cnn_1d_smo_lvlib('train', 'v3+v4', j, k, False, 200, i, 3)
            # acc[i-1] = cnn_2d_smo_lvlib('train', 'v2-v9', j, k, False, 200, i, 3)
            print('Mean accuracy for fold {}/{}: {:.1f} ±{:.1f}% for front-end {}'.format(i, 3, 100*np.mean(acc[:i]), 100*np.std(acc[:i]), k))

#  Socio Bee
def cnn_1d_sociobee(epochs, fold, folds):

    # path = 'D:\\PhD\\Datasets\\SocioBee\\UrESC22\\'
    path = 'D:\\PhD\\Datasets\\SocioBee\\UrANP\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    x_nonpolluting, y_nonpolluting = hi.load_audio(path + 'NP.wav', 0)
    x_polluting, y_polluting = hi.load_audio(path + 'P.wav', 1)
    # np.random.shuffle(x_polluting)
    # np.random.shuffle(x_nonpolluting)

    # make folds
    x_train = np.concatenate((get_array_part(x_polluting, 0, s1),
                              get_array_part(x_nonpolluting, 0, s1),
                              get_array_part(x_polluting, s2, 1),
                              get_array_part(x_nonpolluting, s2, 1)), axis=0)
    y_train = np.concatenate((get_array_part(y_polluting, 0, s1),
                              get_array_part(y_nonpolluting, 0, s1),
                              get_array_part(y_polluting, s2, 1),
                              get_array_part(y_nonpolluting, s2, 1)), axis=0)
    x_test = np.concatenate((get_array_part(x_polluting, s1, s2),
                             get_array_part(x_nonpolluting, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y_polluting, s1, s2),
                             get_array_part(y_nonpolluting, s1, s2)), axis=0)

    score = cnn1d.train(x_train, y_train, x_test, y_test, epochs)

    # print scores
    print('Fold {} with accuracy: {:.1f}'.format(fold, 100 * score))
    return score
for i in range(1, 4):  # folds
    if 'acc' not in locals(): acc = np.zeros((3))
    acc[i-1] = cnn_1d_sociobee(200, i, 3)
    print('Mean accuracy for fold {}/{}: {:.1f} ±{:.1f}%'.format(i, 3, 100*np.mean(acc[:i]), 100*np.std(acc[:i])))

# UrbanSound8k
def ann_esr(mode, epochs, fold, folds):

    # define folds
    s1 = round((folds - fold) * 1.0 / folds, 1)
    s2 = round(s1 + 1.0 / folds, 1)

    # load data
    if mode == 'rnn':
        # load data
        path = 'D:\\PhD\\Datasets\\UrbanSound8k\\legacy\\STFV\\'
        print('Data from:', path)
        for s in ["mfccs"]:
            x_0, y_0 = hi.load_features_csv(path + '_air_conditioner.wav.norm-' + s + '.csv', 0)
            x_1, y_1 = hi.load_features_csv(path + '_car_horn.wav.norm-' + s + '.csv', 1)
            x_2, y_2 = hi.load_features_csv(path + '_children_playing.wav.norm-' + s + '.csv', 2)
            x_3, y_3 = hi.load_features_csv(path + '_dog_bark.wav.norm-' + s + '.csv', 3)
            x_4, y_4 = hi.load_features_csv(path + '_drilling.wav.norm-' + s + '.csv', 4)
            x_5, y_5 = hi.load_features_csv(path + '_engine_idling.wav.norm-' + s + '.csv', 5)
            x_6, y_6 = hi.load_features_csv(path + '_gun_shot.wav.norm-' + s + '.csv', 6)
            x_7, y_7 = hi.load_features_csv(path + '_jackhammer.wav.norm-' + s + '.csv', 7)
            x_8, y_8 = hi.load_features_csv(path + '_siren.wav.norm-' + s + '.csv', 8)
            x_9, y_9 = hi.load_features_csv(path + '_street_music.wav.norm-' + s + '.csv', 9)
        for s in ["p-sha", "p-spr", "s-dec", "s-fla", "s-flu", "s-rol", "s-sha", "s-slo", "s-var", "t-zcr"]:
            kx_0, ky_0 = hi.load_features_csv(path + '_air_conditioner.wav.norm-' + s + '.csv', 0)
            kx_1, ky_1 = hi.load_features_csv(path + '_car_horn.wav.norm-' + s + '.csv', 1)
            kx_2, ky_2 = hi.load_features_csv(path + '_children_playing.wav.norm-' + s + '.csv', 2)
            kx_3, ky_3 = hi.load_features_csv(path + '_dog_bark.wav.norm-' + s + '.csv', 3)
            kx_4, ky_4 = hi.load_features_csv(path + '_drilling.wav.norm-' + s + '.csv', 4)
            kx_5, ky_5 = hi.load_features_csv(path + '_engine_idling.wav.norm-' + s + '.csv', 5)
            kx_6, ky_6 = hi.load_features_csv(path + '_gun_shot.wav.norm-' + s + '.csv', 6)
            kx_7, ky_7 = hi.load_features_csv(path + '_jackhammer.wav.norm-' + s + '.csv', 7)
            kx_8, ky_8 = hi.load_features_csv(path + '_siren.wav.norm-' + s + '.csv', 8)
            kx_9, ky_9 = hi.load_features_csv(path + '_street_music.wav.norm-' + s + '.csv', 9)
            x_0 = np.concatenate((x_0, kx_0), axis=1)
            x_1 = np.concatenate((x_1, kx_1), axis=1)
            x_2 = np.concatenate((x_2, kx_2), axis=1)
            x_3 = np.concatenate((x_3, kx_3), axis=1)
            x_4 = np.concatenate((x_4, kx_4), axis=1)
            x_5 = np.concatenate((x_5, kx_5), axis=1)
            x_6 = np.concatenate((x_6, kx_6), axis=1)
            x_7 = np.concatenate((x_7, kx_7), axis=1)
            x_8 = np.concatenate((x_8, kx_8), axis=1)
            x_9 = np.concatenate((x_9, kx_9), axis=1)
        x_train = np.concatenate((get_array_part(x_0, 0.0, s1),
                                  get_array_part(x_1, 0.0, s1),
                                  get_array_part(x_2, 0.0, s1),
                                  get_array_part(x_3, 0.0, s1),
                                  get_array_part(x_4, 0.0, s1),
                                  get_array_part(x_5, 0.0, s1),
                                  get_array_part(x_6, 0.0, s1),
                                  get_array_part(x_7, 0.0, s1),
                                  get_array_part(x_8, 0.0, s1),
                                  get_array_part(x_9, 0.0, s1),
                                  get_array_part(x_0, s2, 1.0),
                                  get_array_part(x_1, s2, 1.0),
                                  get_array_part(x_2, s2, 1.0),
                                  get_array_part(x_3, s2, 1.0),
                                  get_array_part(x_4, s2, 1.0),
                                  get_array_part(x_5, s2, 1.0),
                                  get_array_part(x_6, s2, 1.0),
                                  get_array_part(x_7, s2, 1.0),
                                  get_array_part(x_8, s2, 1.0),
                                  get_array_part(x_9, s2, 1.0)), axis=0)
        y_train = np.concatenate((get_array_part(y_0, 0.0, s1),
                                  get_array_part(y_1, 0.0, s1),
                                  get_array_part(y_2, 0.0, s1),
                                  get_array_part(y_3, 0.0, s1),
                                  get_array_part(y_4, 0.0, s1),
                                  get_array_part(y_5, 0.0, s1),
                                  get_array_part(y_6, 0.0, s1),
                                  get_array_part(y_7, 0.0, s1),
                                  get_array_part(y_8, 0.0, s1),
                                  get_array_part(y_9, 0.0, s1),
                                  get_array_part(y_0, s2, 1.0),
                                  get_array_part(y_1, s2, 1.0),
                                  get_array_part(y_2, s2, 1.0),
                                  get_array_part(y_3, s2, 1.0),
                                  get_array_part(y_4, s2, 1.0),
                                  get_array_part(y_5, s2, 1.0),
                                  get_array_part(y_6, s2, 1.0),
                                  get_array_part(y_7, s2, 1.0),
                                  get_array_part(y_8, s2, 1.0),
                                  get_array_part(y_9, s2, 1.0)), axis=0)
        x_test = np.concatenate((get_array_part(x_0, s1, s2),
                                 get_array_part(x_1, s1, s2),
                                 get_array_part(x_2, s1, s2),
                                 get_array_part(x_3, s1, s2),
                                 get_array_part(x_4, s1, s2),
                                 get_array_part(x_5, s1, s2),
                                 get_array_part(x_6, s1, s2),
                                 get_array_part(x_7, s1, s2),
                                 get_array_part(x_8, s1, s2),
                                 get_array_part(x_9, s1, s2)), axis=0)
        y_test = np.concatenate((get_array_part(y_0, s1, s2),
                                 get_array_part(y_1, s1, s2),
                                 get_array_part(y_2, s1, s2),
                                 get_array_part(y_3, s1, s2),
                                 get_array_part(y_4, s1, s2),
                                 get_array_part(y_5, s1, s2),
                                 get_array_part(y_6, s1, s2),
                                 get_array_part(y_7, s1, s2),
                                 get_array_part(y_8, s1, s2),
                                 get_array_part(y_9, s1, s2)), axis=0)

        # remove unwanted features
        r = [0]
        print('Number of initial features: {}'.format(x_train.shape[1]))
        print('Mode: {} excluding features: {}'.format(mode, r))
        x_train = np.delete(x_train, r, axis=1)
        x_test = np.delete(x_test, r, axis=1)
        print('Number of intermediate features: {}'.format(x_train.shape[1]))

        # standardize
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        # reform data to sequential
        ts_length = 128
        ts_step = 64
        ts_data = np.zeros((int(x_train.shape[0] / ts_step), ts_length, x_train.shape[1]))
        ts_truth = np.zeros(int(y_train.shape[0] / ts_step))
        for i in range(0, x_train.shape[0] - ts_length, ts_step):
            ts_data[int(i / ts_step), :, :] = x_train[i:i + ts_length, :]
            ts_truth[int(i / ts_step)] = y_train[i]
        x_train = ts_data
        y_train = ts_truth
        ts_data = np.zeros((int(x_test.shape[0] / ts_step), ts_length, x_test.shape[1]))
        ts_truth = np.zeros(int(y_test.shape[0] / ts_step))
        for i in range(0, x_test.shape[0] - ts_length, ts_step):
            ts_data[int(i / ts_step), :, :] = x_test[i:i + ts_length, :]
            ts_truth[int(i / ts_step)] = y_test[i]
        x_test = ts_data
        y_test = ts_truth

        # train
        score = annlstm.train(x_train, y_train, x_test, y_test, 10, epochs)
    else:
        path = 'D:\\PhD\\Datasets\\UrbanSound8k\\legacy\\ADi\\'
        x_0, y_0 = hi.load_features_csv(path + '_air_conditioner.csv', 0)
        x_1, y_1 = hi.load_features_csv(path + '_car_horn.csv', 1)
        x_2, y_2 = hi.load_features_csv(path + '_children_playing.csv', 2)
        x_3, y_3 = hi.load_features_csv(path + '_dog_bark.csv', 3)
        x_4, y_4 = hi.load_features_csv(path + '_drilling.csv', 4)
        x_5, y_5 = hi.load_features_csv(path + '_engine_idling.csv', 5)
        x_6, y_6 = hi.load_features_csv(path + '_gun_shot.csv', 6)
        x_7, y_7 = hi.load_features_csv(path + '_jackhammer.csv', 7)
        x_8, y_8 = hi.load_features_csv(path + '_siren.csv', 8)
        x_9, y_9 = hi.load_features_csv(path + '_street_music.csv', 9)

        # legacy simple
        if False:
            x_train = np.concatenate((get_array_part(x_0, 0.0, s1),
                                      get_array_part(x_1, 0.0, s1),
                                      get_array_part(x_2, 0.0, s1),
                                      get_array_part(x_3, 0.0, s1),
                                      get_array_part(x_4, 0.0, s1),
                                      get_array_part(x_5, 0.0, s1),
                                      get_array_part(x_6, 0.0, s1),
                                      get_array_part(x_7, 0.0, s1),
                                      get_array_part(x_8, 0.0, s1),
                                      get_array_part(x_9, 0.0, s1),
                                      get_array_part(x_0, s2, 1.0),
                                      get_array_part(x_1, s2, 1.0),
                                      get_array_part(x_2, s2, 1.0),
                                      get_array_part(x_3, s2, 1.0),
                                      get_array_part(x_4, s2, 1.0),
                                      get_array_part(x_5, s2, 1.0),
                                      get_array_part(x_6, s2, 1.0),
                                      get_array_part(x_7, s2, 1.0),
                                      get_array_part(x_8, s2, 1.0),
                                      get_array_part(x_9, s2, 1.0)), axis=0)
            y_train = np.concatenate((get_array_part(y_0, 0.0, s1),
                                      get_array_part(y_1, 0.0, s1),
                                      get_array_part(y_2, 0.0, s1),
                                      get_array_part(y_3, 0.0, s1),
                                      get_array_part(y_4, 0.0, s1),
                                      get_array_part(y_5, 0.0, s1),
                                      get_array_part(y_6, 0.0, s1),
                                      get_array_part(y_7, 0.0, s1),
                                      get_array_part(y_8, 0.0, s1),
                                      get_array_part(y_9, 0.0, s1),
                                      get_array_part(y_0, s2, 1.0),
                                      get_array_part(y_1, s2, 1.0),
                                      get_array_part(y_2, s2, 1.0),
                                      get_array_part(y_3, s2, 1.0),
                                      get_array_part(y_4, s2, 1.0),
                                      get_array_part(y_5, s2, 1.0),
                                      get_array_part(y_6, s2, 1.0),
                                      get_array_part(y_7, s2, 1.0),
                                      get_array_part(y_8, s2, 1.0),
                                      get_array_part(y_9, s2, 1.0)), axis=0)
            x_test = np.concatenate((get_array_part(x_0, s1, s2),
                                     get_array_part(x_1, s1, s2),
                                     get_array_part(x_2, s1, s2),
                                     get_array_part(x_3, s1, s2),
                                     get_array_part(x_4, s1, s2),
                                     get_array_part(x_5, s1, s2),
                                     get_array_part(x_6, s1, s2),
                                     get_array_part(x_7, s1, s2),
                                     get_array_part(x_8, s1, s2),
                                     get_array_part(x_9, s1, s2)), axis=0)
            y_test = np.concatenate((get_array_part(y_0, s1, s2),
                                     get_array_part(y_1, s1, s2),
                                     get_array_part(y_2, s1, s2),
                                     get_array_part(y_3, s1, s2),
                                     get_array_part(y_4, s1, s2),
                                     get_array_part(y_5, s1, s2),
                                     get_array_part(y_6, s1, s2),
                                     get_array_part(y_7, s1, s2),
                                     get_array_part(y_8, s1, s2),
                                     get_array_part(y_9, s1, s2)), axis=0)

        # legacy complex
        if True:
            air = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
            car = [0.00, 0.06, 0.15, 0.25, 0.41, 0.72, 0.77, 0.82, 0.88, 0.92, 1.00]
            children = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
            dog = [0.00, 0.10, 0.19, 0.30, 0.39, 0.49, 0.59, 0.70, 0.80, 0.90, 1.00]
            drilling = [0.00, 0.10, 0.19, 0.29, 0.40, 0.49, 0.59, 0.70, 0.80, 0.90, 1.00]
            engine = [0.00, 0.10, 0.20, 0.30, 0.41, 0.52, 0.63, 0.73, 0.82, 0.91, 1.00]
            gun = [0.00, 0.11, 0.20, 0.29, 0.40, 0.48, 0.58, 0.74, 0.81, 0.91, 1.00]
            jackhammer = [0.00, 0.12, 0.24, 0.37, 0.47, 0.59, 0.67, 0.74, 0.82, 0.90, 1.00]
            siren = [0.00, 0.09, 0.19, 0.33, 0.50, 0.58, 0.66, 0.74, 0.82, 0.91, 1.00]
            street = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

            x_train = np.concatenate((get_array_part(x_0, 0.0, air[fold-1]),
                                      get_array_part(x_1, 0.0, car[fold-1]),
                                      get_array_part(x_2, 0.0, children[fold-1]),
                                      get_array_part(x_3, 0.0, dog[fold-1]),
                                      get_array_part(x_4, 0.0, drilling[fold-1]),
                                      get_array_part(x_5, 0.0, engine[fold-1]),
                                      get_array_part(x_6, 0.0, gun[fold-1]),
                                      get_array_part(x_7, 0.0, jackhammer[fold-1]),
                                      get_array_part(x_8, 0.0, siren[fold-1]),
                                      get_array_part(x_9, 0.0, street[fold-1]),
                                      get_array_part(x_0, air[fold], 1.0),
                                      get_array_part(x_1, car[fold], 1.0),
                                      get_array_part(x_2, children[fold], 1.0),
                                      get_array_part(x_3, dog[fold], 1.0),
                                      get_array_part(x_4, drilling[fold], 1.0),
                                      get_array_part(x_5, engine[fold], 1.0),
                                      get_array_part(x_6, gun[fold], 1.0),
                                      get_array_part(x_7, jackhammer[fold], 1.0),
                                      get_array_part(x_8, siren[fold], 1.0),
                                      get_array_part(x_9, street[fold], 1.0)), axis=0)
            y_train = np.concatenate((get_array_part(y_0, 0.0, air[fold-1]),
                                      get_array_part(y_1, 0.0, car[fold-1]),
                                      get_array_part(y_2, 0.0, children[fold-1]),
                                      get_array_part(y_3, 0.0, dog[fold-1]),
                                      get_array_part(y_4, 0.0, drilling[fold-1]),
                                      get_array_part(y_5, 0.0, engine[fold-1]),
                                      get_array_part(y_6, 0.0, gun[fold-1]),
                                      get_array_part(y_7, 0.0, jackhammer[fold-1]),
                                      get_array_part(y_8, 0.0, siren[fold-1]),
                                      get_array_part(y_9, 0.0, street[fold-1]),
                                      get_array_part(y_0, air[fold], 1.0),
                                      get_array_part(y_1, car[fold], 1.0),
                                      get_array_part(y_2, children[fold], 1.0),
                                      get_array_part(y_3, dog[fold], 1.0),
                                      get_array_part(y_4, drilling[fold], 1.0),
                                      get_array_part(y_5, engine[fold], 1.0),
                                      get_array_part(y_6, gun[fold], 1.0),
                                      get_array_part(y_7, jackhammer[fold], 1.0),
                                      get_array_part(y_8, siren[fold], 1.0),
                                      get_array_part(y_9, street[fold], 1.0)), axis=0)
            x_test = np.concatenate((get_array_part(x_0, air[fold-1], air[fold]),
                                      get_array_part(x_1, car[fold-1], car[fold]),
                                      get_array_part(x_2, children[fold-1], children[fold]),
                                      get_array_part(x_3, dog[fold-1], dog[fold]),
                                      get_array_part(x_4, drilling[fold-1], drilling[fold]),
                                      get_array_part(x_5, engine[fold-1], engine[fold]),
                                      get_array_part(x_6, gun[fold-1], gun[fold]),
                                      get_array_part(x_7, jackhammer[fold-1], jackhammer[fold]),
                                      get_array_part(x_8, siren[fold-1], siren[fold]),
                                      get_array_part(x_9, street[fold-1], street[fold])), axis=0)
            y_test = np.concatenate((get_array_part(y_0, air[fold-1], air[fold]),
                                      get_array_part(y_1, car[fold-1], car[fold]),
                                      get_array_part(y_2, children[fold-1], children[fold]),
                                      get_array_part(y_3, dog[fold-1], dog[fold]),
                                      get_array_part(y_4, drilling[fold-1], drilling[fold]),
                                      get_array_part(y_5, engine[fold-1], engine[fold]),
                                      get_array_part(y_6, gun[fold-1], gun[fold]),
                                      get_array_part(y_7, jackhammer[fold-1], jackhammer[fold]),
                                      get_array_part(y_8, siren[fold-1], siren[fold]),
                                      get_array_part(y_9, street[fold-1], street[fold])), axis=0)

        # remove unwanted features
        r = []
        #r.extend([288,289,290,291,292,293,294,295,296,297,298,299])
        if mode == 'sti':
            for i in range(x_train.shape[1]):
                if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                    r.append(i)
            r.append(0)
        elif mode == 'adi':
            for i in range(x_train.shape[1]):
                if i % 12 == 0 or i % 12 == 1 or i % 12 == 2 or i % 12 == 3 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                    r.append(i)
            r.append(7)
        elif mode == 'sti-eti':
            for i in range(x_train.shape[1]):
                if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7:
                    r.append(i)
            r.append(0)
        elif mode == 'adi-eti':
            for i in range(x_train.shape[1]):
                if i % 12 == 0 or i % 12 == 1 or i % 12 == 4 or i % 12 == 5:
                    r.append(i)
            r.append(7)
        print('Number of initial features: {}'.format(x_train.shape[1]))
        #print('Mode: {} excluding features: {}'.format(mode, r))
        x_train = np.delete(x_train, r, axis=1)
        x_test = np.delete(x_test, r, axis=1)
        print('Number of intermediate features: {}'.format(x_train.shape[1]))

        # feature selection
        '''selector = SelectFromModel(DecisionTreeClassifier(), threshold=-np.inf, max_features=min(125, x_train_lvlib.shape[1]))
        selector = GenericUnivariateSelect(f_classif, mode='fwe') # fpr, fdr, fwe, ('k_best', param=100) mutual_info_classif f_classif
        selector.fit(x_train_lvlib, y_train_lvlib)
        x_train_lvlib = selector.transform(x_train_lvlib)
        x_test_lvlib = selector.transform(x_test_lvlib)
        print('Number of selected features: {}'.format(x_train_lvlib.shape[1]))'''

        # outlier filtering
        '''z = np.abs(stats.zscore(x_train_lvlib))
        x_train_lvlib = x_train_lvlib[(z < 10).all(axis=1)]
        y_train_lvlib = y_train_lvlib[(z < 10).all(axis=1)]'''

        # standardize
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        # train
        score = ann.train(x_train, y_train, x_test, y_test, epochs)
        #score = LogisticRegression(random_state=0, multi_class='auto', solver='liblinear').fit(x_train_lvlib, y_train_lvlib).score(x_test_lvlib, y_test)
        #score = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (55, 55), random_state = 1).fit(x_train_lvlib, y_train_lvlib).score(x_test_lvlib, y_test)

    print('Fold {}, mode {}, accuracy: {}'.format(fold, mode, round(100*score,1)))
    return score
def cnn_1d_esr(mode, normalize, lstm, epochs, fold, folds):

    path = 'D:\\PhD\\Datasets\\UrbanSound8K\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load and form data
    if lstm:
        x_0, y_0 = hi.load_audio_ts(path + '_air_conditioner.wav', 0)
        x_1, y_1 = hi.load_audio_ts(path + '_car_horn.wav', 1)
        x_2, y_2 = hi.load_audio_ts(path + '_children_playing.wav', 2)
        x_3, y_3 = hi.load_audio_ts(path + '_dog_bark.wav', 3)
        x_4, y_4 = hi.load_audio_ts(path + '_drilling.wav', 4)
        x_5, y_5 = hi.load_audio_ts(path + '_engine_idling.wav', 5)
        x_6, y_6 = hi.load_audio_ts(path + '_gun_shot.wav', 6)
        x_7, y_7 = hi.load_audio_ts(path + '_jackhammer.wav', 7)
        x_8, y_8 = hi.load_audio_ts(path + '_siren.wav', 8)
        x_9, y_9 = hi.load_audio_ts(path + '_street_music.wav', 9)
    else:
        x_0, y_0 = hi.load_audio(path + '_air_conditioner.wav', 0)
        x_1, y_1 = hi.load_audio(path + '_car_horn.wav', 1)
        x_2, y_2 = hi.load_audio(path + '_children_playing.wav', 2)
        x_3, y_3 = hi.load_audio(path + '_dog_bark.wav', 3)
        x_4, y_4 = hi.load_audio(path + '_drilling.wav', 4)
        x_5, y_5 = hi.load_audio(path + '_engine_idling.wav', 5)
        x_6, y_6 = hi.load_audio(path + '_gun_shot.wav', 6)
        x_7, y_7 = hi.load_audio(path + '_jackhammer.wav', 7)
        x_8, y_8 = hi.load_audio(path + '_siren.wav', 8)
        x_9, y_9 = hi.load_audio(path + '_street_music.wav', 9)

    x_train = np.concatenate((get_array_part(x_0, 0.0, s1),
                              get_array_part(x_1, 0.0, s1),
                              get_array_part(x_2, 0.0, s1),
                              get_array_part(x_3, 0.0, s1),
                              get_array_part(x_4, 0.0, s1),
                              get_array_part(x_5, 0.0, s1),
                              get_array_part(x_6, 0.0, s1),
                              get_array_part(x_7, 0.0, s1),
                              get_array_part(x_8, 0.0, s1),
                              get_array_part(x_9, 0.0, s1),
                              get_array_part(x_0, s2, 1.0),
                              get_array_part(x_1, s2, 1.0),
                              get_array_part(x_2, s2, 1.0),
                              get_array_part(x_3, s2, 1.0),
                              get_array_part(x_4, s2, 1.0),
                              get_array_part(x_5, s2, 1.0),
                              get_array_part(x_6, s2, 1.0),
                              get_array_part(x_7, s2, 1.0),
                              get_array_part(x_8, s2, 1.0),
                              get_array_part(x_9, s2, 1.0)), axis=0)
    y_train = np.concatenate((get_array_part(y_0, 0.0, s1),
                              get_array_part(y_1, 0.0, s1),
                              get_array_part(y_2, 0.0, s1),
                              get_array_part(y_3, 0.0, s1),
                              get_array_part(y_4, 0.0, s1),
                              get_array_part(y_5, 0.0, s1),
                              get_array_part(y_6, 0.0, s1),
                              get_array_part(y_7, 0.0, s1),
                              get_array_part(y_8, 0.0, s1),
                              get_array_part(y_9, 0.0, s1),
                              get_array_part(y_0, s2, 1.0),
                              get_array_part(y_1, s2, 1.0),
                              get_array_part(y_2, s2, 1.0),
                              get_array_part(y_3, s2, 1.0),
                              get_array_part(y_4, s2, 1.0),
                              get_array_part(y_5, s2, 1.0),
                              get_array_part(y_6, s2, 1.0),
                              get_array_part(y_7, s2, 1.0),
                              get_array_part(y_8, s2, 1.0),
                              get_array_part(y_9, s2, 1.0)), axis=0)
    x_test = np.concatenate((get_array_part(x_0, s1, s2),
                              get_array_part(x_1, s1, s2),
                              get_array_part(x_2, s1, s2),
                              get_array_part(x_3, s1, s2),
                              get_array_part(x_4, s1, s2),
                              get_array_part(x_5, s1, s2),
                              get_array_part(x_6, s1, s2),
                              get_array_part(x_7, s1, s2),
                              get_array_part(x_8, s1, s2),
                              get_array_part(x_9, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y_0, s1, s2),
                              get_array_part(y_1, s1, s2),
                              get_array_part(y_2, s1, s2),
                              get_array_part(y_3, s1, s2),
                              get_array_part(y_4, s1, s2),
                              get_array_part(y_5, s1, s2),
                              get_array_part(y_6, s1, s2),
                              get_array_part(y_7, s1, s2),
                              get_array_part(y_8, s1, s2),
                              get_array_part(y_9, s1, s2)), axis=0)

    if lstm:
        #
        score = cnn1dlstm.train(x_train, y_train, x_test, y_test, 10, epochs)
    else:
        if mode == 'train':
            score = cnn1d.train(x_train, y_train, x_test, y_test, epochs)
        elif mode == 'retrain':
            score = cnn1d.retrain(x_train, y_train, x_test, y_test, epochs)
        else:
            score = -1

    print('Fold {} with accuracy: {:.1f}'.format(fold, 100*score))
    return score
def cnn_2d_esr(mode, normalize, lstm, epochs, fold, folds):
    print('\n*****************************************************************')
    print('\nCNN2D ESR\n')

    # legacy simple
    if False:
        path = 'D:\\PhD\\Datasets\\UrbanSound8K\\legacy\\MEL\\'
        s1 = round((folds - fold) * 1.0 / folds, 2)
        s2 = round(s1 + 1.0 / folds, 2)
        print('Data from:', path)
        if lstm:
            x_0, y_0 = hi.load_spectrum_csv_ts(path + '_air_conditioner.csv', 0)
            x_1, y_1 = hi.load_spectrum_csv_ts(path + '_car_horn.csv', 1)
            x_2, y_2 = hi.load_spectrum_csv_ts(path + '_children_playing.csv', 2)
            x_3, y_3 = hi.load_spectrum_csv_ts(path + '_dog_bark.csv', 3)
            x_4, y_4 = hi.load_spectrum_csv_ts(path + '_drilling.csv', 4)
            x_5, y_5 = hi.load_spectrum_csv_ts(path + '_engine_idling.csv', 5)
            x_6, y_6 = hi.load_spectrum_csv_ts(path + '_gun_shot.csv', 6)
            x_7, y_7 = hi.load_spectrum_csv_ts(path + '_jackhammer.csv', 7)
            x_8, y_8 = hi.load_spectrum_csv_ts(path + '_siren.csv', 8)
            x_9, y_9 = hi.load_spectrum_csv_ts(path + '_street_music.csv', 9)
        else:
            x_0, y_0 = hi.load_spectrum_csv(path + '_air_conditioner.csv', 0)
            x_1, y_1 = hi.load_spectrum_csv(path + '_car_horn.csv', 1)
            x_2, y_2 = hi.load_spectrum_csv(path + '_children_playing.csv', 2)
            x_3, y_3 = hi.load_spectrum_csv(path + '_dog_bark.csv', 3)
            x_4, y_4 = hi.load_spectrum_csv(path + '_drilling.csv', 4)
            x_5, y_5 = hi.load_spectrum_csv(path + '_engine_idling.csv', 5)
            x_6, y_6 = hi.load_spectrum_csv(path + '_gun_shot.csv', 6)
            x_7, y_7 = hi.load_spectrum_csv(path + '_jackhammer.csv', 7)
            x_8, y_8 = hi.load_spectrum_csv(path + '_siren.csv', 8)
            x_9, y_9 = hi.load_spectrum_csv(path + '_street_music.csv', 9)
        x_train = np.concatenate((get_array_part(x_0, 0.0, s1),
                                  get_array_part(x_1, 0.0, s1),
                                  get_array_part(x_2, 0.0, s1),
                                  get_array_part(x_3, 0.0, s1),
                                  get_array_part(x_4, 0.0, s1),
                                  get_array_part(x_5, 0.0, s1),
                                  get_array_part(x_6, 0.0, s1),
                                  get_array_part(x_7, 0.0, s1),
                                  get_array_part(x_8, 0.0, s1),
                                  get_array_part(x_9, 0.0, s1),
                                  get_array_part(x_0, s2, 1.0),
                                  get_array_part(x_1, s2, 1.0),
                                  get_array_part(x_2, s2, 1.0),
                                  get_array_part(x_3, s2, 1.0),
                                  get_array_part(x_4, s2, 1.0),
                                  get_array_part(x_5, s2, 1.0),
                                  get_array_part(x_6, s2, 1.0),
                                  get_array_part(x_7, s2, 1.0),
                                  get_array_part(x_8, s2, 1.0),
                                  get_array_part(x_9, s2, 1.0)), axis=0)
        y_train = np.concatenate((get_array_part(y_0, 0.0, s1),
                                  get_array_part(y_1, 0.0, s1),
                                  get_array_part(y_2, 0.0, s1),
                                  get_array_part(y_3, 0.0, s1),
                                  get_array_part(y_4, 0.0, s1),
                                  get_array_part(y_5, 0.0, s1),
                                  get_array_part(y_6, 0.0, s1),
                                  get_array_part(y_7, 0.0, s1),
                                  get_array_part(y_8, 0.0, s1),
                                  get_array_part(y_9, 0.0, s1),
                                  get_array_part(y_0, s2, 1.0),
                                  get_array_part(y_1, s2, 1.0),
                                  get_array_part(y_2, s2, 1.0),
                                  get_array_part(y_3, s2, 1.0),
                                  get_array_part(y_4, s2, 1.0),
                                  get_array_part(y_5, s2, 1.0),
                                  get_array_part(y_6, s2, 1.0),
                                  get_array_part(y_7, s2, 1.0),
                                  get_array_part(y_8, s2, 1.0),
                                  get_array_part(y_9, s2, 1.0)), axis=0)
        x_test = np.concatenate((get_array_part(x_0, s1, s2),
                                  get_array_part(x_1, s1, s2),
                                  get_array_part(x_2, s1, s2),
                                  get_array_part(x_3, s1, s2),
                                  get_array_part(x_4, s1, s2),
                                  get_array_part(x_5, s1, s2),
                                  get_array_part(x_6, s1, s2),
                                  get_array_part(x_7, s1, s2),
                                  get_array_part(x_8, s1, s2),
                                  get_array_part(x_9, s1, s2)), axis=0)
        y_test = np.concatenate((get_array_part(y_0, s1, s2),
                                  get_array_part(y_1, s1, s2),
                                  get_array_part(y_2, s1, s2),
                                  get_array_part(y_3, s1, s2),
                                  get_array_part(y_4, s1, s2),
                                  get_array_part(y_5, s1, s2),
                                  get_array_part(y_6, s1, s2),
                                  get_array_part(y_7, s1, s2),
                                  get_array_part(y_8, s1, s2),
                                  get_array_part(y_9, s1, s2)), axis=0)

    # legacy complex
    if False:
        path = 'D:\\PhD\\Datasets\\UrbanSound8K\\legacy\\MEL\\'
        print('Data from:', path)
        if lstm:
            x_0, y_0 = hi.load_spectrum_csv_ts(path + '_air_conditioner.csv', 0)
            x_1, y_1 = hi.load_spectrum_csv_ts(path + '_car_horn.csv', 1)
            x_2, y_2 = hi.load_spectrum_csv_ts(path + '_children_playing.csv', 2)
            x_3, y_3 = hi.load_spectrum_csv_ts(path + '_dog_bark.csv', 3)
            x_4, y_4 = hi.load_spectrum_csv_ts(path + '_drilling.csv', 4)
            x_5, y_5 = hi.load_spectrum_csv_ts(path + '_engine_idling.csv', 5)
            x_6, y_6 = hi.load_spectrum_csv_ts(path + '_gun_shot.csv', 6)
            x_7, y_7 = hi.load_spectrum_csv_ts(path + '_jackhammer.csv', 7)
            x_8, y_8 = hi.load_spectrum_csv_ts(path + '_siren.csv', 8)
            x_9, y_9 = hi.load_spectrum_csv_ts(path + '_street_music.csv', 9)
        else:
            x_0, y_0 = hi.load_spectrum_csv(path + '_air_conditioner.csv', 0)
            x_1, y_1 = hi.load_spectrum_csv(path + '_car_horn.csv', 1)
            x_2, y_2 = hi.load_spectrum_csv(path + '_children_playing.csv', 2)
            x_3, y_3 = hi.load_spectrum_csv(path + '_dog_bark.csv', 3)
            x_4, y_4 = hi.load_spectrum_csv(path + '_drilling.csv', 4)
            x_5, y_5 = hi.load_spectrum_csv(path + '_engine_idling.csv', 5)
            x_6, y_6 = hi.load_spectrum_csv(path + '_gun_shot.csv', 6)
            x_7, y_7 = hi.load_spectrum_csv(path + '_jackhammer.csv', 7)
            x_8, y_8 = hi.load_spectrum_csv(path + '_siren.csv', 8)
            x_9, y_9 = hi.load_spectrum_csv(path + '_street_music.csv', 9)
        air = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        car = [0.00, 0.06, 0.15, 0.25, 0.41, 0.72, 0.77, 0.82, 0.88, 0.92, 1.00]
        children = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        dog = [0.00, 0.10, 0.19, 0.30, 0.39, 0.49, 0.59, 0.70, 0.80, 0.90, 1.00]
        drilling = [0.00, 0.10, 0.19, 0.29, 0.40, 0.49, 0.59, 0.70, 0.80, 0.90, 1.00]
        engine = [0.00, 0.10, 0.20, 0.30, 0.41, 0.52, 0.63, 0.73, 0.82, 0.91, 1.00]
        gun = [0.00, 0.11, 0.20, 0.29, 0.40, 0.48, 0.58, 0.74, 0.81, 0.91, 1.00]
        jackhammer = [0.00, 0.12, 0.24, 0.37, 0.47, 0.59, 0.67, 0.74, 0.82, 0.90, 1.00]
        siren = [0.00, 0.09, 0.19, 0.33, 0.50, 0.58, 0.66, 0.74, 0.82, 0.91, 1.00]
        street = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        x_train = np.concatenate((get_array_part(x_0, 0.0, air[fold - 1]),
                                  get_array_part(x_1, 0.0, car[fold - 1]),
                                  get_array_part(x_2, 0.0, children[fold - 1]),
                                  get_array_part(x_3, 0.0, dog[fold - 1]),
                                  get_array_part(x_4, 0.0, drilling[fold - 1]),
                                  get_array_part(x_5, 0.0, engine[fold - 1]),
                                  get_array_part(x_6, 0.0, gun[fold - 1]),
                                  get_array_part(x_7, 0.0, jackhammer[fold - 1]),
                                  get_array_part(x_8, 0.0, siren[fold - 1]),
                                  get_array_part(x_9, 0.0, street[fold - 1]),
                                  get_array_part(x_0, air[fold], 1.0),
                                  get_array_part(x_1, car[fold], 1.0),
                                  get_array_part(x_2, children[fold], 1.0),
                                  get_array_part(x_3, dog[fold], 1.0),
                                  get_array_part(x_4, drilling[fold], 1.0),
                                  get_array_part(x_5, engine[fold], 1.0),
                                  get_array_part(x_6, gun[fold], 1.0),
                                  get_array_part(x_7, jackhammer[fold], 1.0),
                                  get_array_part(x_8, siren[fold], 1.0),
                                  get_array_part(x_9, street[fold], 1.0)), axis=0)
        y_train = np.concatenate((get_array_part(y_0, 0.0, air[fold - 1]),
                                  get_array_part(y_1, 0.0, car[fold - 1]),
                                  get_array_part(y_2, 0.0, children[fold - 1]),
                                  get_array_part(y_3, 0.0, dog[fold - 1]),
                                  get_array_part(y_4, 0.0, drilling[fold - 1]),
                                  get_array_part(y_5, 0.0, engine[fold - 1]),
                                  get_array_part(y_6, 0.0, gun[fold - 1]),
                                  get_array_part(y_7, 0.0, jackhammer[fold - 1]),
                                  get_array_part(y_8, 0.0, siren[fold - 1]),
                                  get_array_part(y_9, 0.0, street[fold - 1]),
                                  get_array_part(y_0, air[fold], 1.0),
                                  get_array_part(y_1, car[fold], 1.0),
                                  get_array_part(y_2, children[fold], 1.0),
                                  get_array_part(y_3, dog[fold], 1.0),
                                  get_array_part(y_4, drilling[fold], 1.0),
                                  get_array_part(y_5, engine[fold], 1.0),
                                  get_array_part(y_6, gun[fold], 1.0),
                                  get_array_part(y_7, jackhammer[fold], 1.0),
                                  get_array_part(y_8, siren[fold], 1.0),
                                  get_array_part(y_9, street[fold], 1.0)), axis=0)
        x_test = np.concatenate((get_array_part(x_0, air[fold - 1], air[fold]),
                                 get_array_part(x_1, car[fold - 1], car[fold]),
                                 get_array_part(x_2, children[fold - 1], children[fold]),
                                 get_array_part(x_3, dog[fold - 1], dog[fold]),
                                 get_array_part(x_4, drilling[fold - 1], drilling[fold]),
                                 get_array_part(x_5, engine[fold - 1], engine[fold]),
                                 get_array_part(x_6, gun[fold - 1], gun[fold]),
                                 get_array_part(x_7, jackhammer[fold - 1], jackhammer[fold]),
                                 get_array_part(x_8, siren[fold - 1], siren[fold]),
                                 get_array_part(x_9, street[fold - 1], street[fold])), axis=0)
        y_test = np.concatenate((get_array_part(y_0, air[fold - 1], air[fold]),
                                 get_array_part(y_1, car[fold - 1], car[fold]),
                                 get_array_part(y_2, children[fold - 1], children[fold]),
                                 get_array_part(y_3, dog[fold - 1], dog[fold]),
                                 get_array_part(y_4, drilling[fold - 1], drilling[fold]),
                                 get_array_part(y_5, engine[fold - 1], engine[fold]),
                                 get_array_part(y_6, gun[fold - 1], gun[fold]),
                                 get_array_part(y_7, jackhammer[fold - 1], jackhammer[fold]),
                                 get_array_part(y_8, siren[fold - 1], siren[fold]),
                                 get_array_part(y_9, street[fold - 1], street[fold])), axis=0)

    # official
    if True:
        path = 'D:\\PhD\\Datasets\\UrbanSound8K\\full\\MEL\\'
        print('Data from:', path)
        for i in range(10):

            # print('Loading fold {:d}'.format(i+1))

            #if i == fold-1:   path = 'D:\\PhD\\Datasets\\UrbanSound8K\\full\\'
            #else:             path = 'D:\\PhD\\Datasets\\UrbanSound8K\\salient\\'

            x_0, y_0 = hi.load_spectrum_csv(path + 'air_conditioner_' + str(i+1) + '.csv', 0, normalize)
            x_1, y_1 = hi.load_spectrum_csv(path + 'car_horn_' + str(i+1) + '.csv', 1, normalize)
            x_2, y_2 = hi.load_spectrum_csv(path + 'children_playing_' + str(i+1) + '.csv', 2, normalize)
            x_3, y_3 = hi.load_spectrum_csv(path + 'dog_bark_'+ str(i+1) + '.csv', 3, normalize)
            x_4, y_4 = hi.load_spectrum_csv(path + 'drilling_'+ str(i+1) + '.csv', 4, normalize)
            x_5, y_5 = hi.load_spectrum_csv(path + 'engine_idling_'+ str(i+1) + '.csv', 5, normalize)
            x_6, y_6 = hi.load_spectrum_csv(path + 'gun_shot_'+ str(i+1) + '.csv', 6, normalize)
            x_7, y_7 = hi.load_spectrum_csv(path + 'jackhammer_'+ str(i+1) + '.csv', 7, normalize)
            x_8, y_8 = hi.load_spectrum_csv(path + 'siren_'+ str(i+1) + '.csv', 8, normalize)
            x_9, y_9 = hi.load_spectrum_csv(path + 'street_music_'+ str(i+1) + '.csv', 9, normalize)

            if i == 0:
                x_train = np.zeros((0, x_0.shape[1], x_0.shape[2]))
                y_train = np.zeros((0, y_0.shape[1]))
                x_test = np.zeros((0, x_0.shape[1], x_0.shape[2]))
                y_test = np.zeros((0, y_0.shape[1]))

            if i == fold-1:
                x_test = np.append(x_test, np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0), axis=0)
                y_test = np.append(y_test, np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0), axis=0)
            else:
                x_train = np.append(x_train, np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0), axis=0)
                y_train = np.append(y_train, np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0), axis=0)

    # train
    if lstm:
        #
        score = cnn2dlstm.train(x_train, y_train, x_test, y_test, epochs)
    else:
        if mode == 'train':
            score = cnn2d.train(x_train, y_train, x_test, y_test, epochs, 5)
            # score = tcnn2d.train(x_train_lvlib, y_train_lvlib, x_test_lvlib, y_test, epochs)
        elif mode == 'retrain':
            score = cnn2d.retrain(x_train, y_train, x_test, y_test, epochs)
        elif mode == 'evaluate:':
            score = cnn2d.evaluate(x_train, y_train, x_test, y_test, epochs)

    print('\nFold {} with accuracy: {:.1f}\n'.format(fold, 100*score))
    return score
#transform('D:\\PhD\\Datasets\\UrbanSound8k\\full\\PCM\\', 'pcen')
for k in []:
    for i in range(1, 11):
        if i==1: acc = np.zeros((10))
        #acc[i-1] = ann_esr('adi-eti', 200, i, 10)               # sti-eti, adi-eti, rnn
        acc[i-1] = cnn_1d_esr('train', False, 200, i, 10)
        # acc[i-1] = cnn_2d_esr('train', k, False, 200, i, 10)
        print('Mean accuracy: {:.1f} ±{:.1f} for front-end {}'.format(100 * np.mean(acc[:i]), 100 * np.std(acc[:i]), k))

# ESC-LVLib
def ann_escbdlib(mode, epochs):

    # information
    print('*****************************************************************')
    print('RNN ESC-BDLib')

    # load and form train data
    path = 'E:\FastDatasets\ESC-BDLib\ADi\\'
    print('\nTrain data from:', path)
    x_0, y_0 = hi.load_features_csv(path + '01_airplanes.csv', 0)
    x_1, y_1 = hi.load_features_csv(path + '02_alarms.csv', 1)
    x_2, y_2 = hi.load_features_csv(path + '03_applause.csv', 2)
    x_3, y_3 = hi.load_features_csv(path + '04_birds.csv', 3)
    x_4, y_4 = hi.load_features_csv(path + '05_dogs.csv', 4)
    x_5, y_5 = hi.load_features_csv(path + '06_motorcycles.csv', 5)
    x_6, y_6 = hi.load_features_csv(path + '07_rain.csv', 6)
    x_7, y_7 = hi.load_features_csv(path + '08_rivers.csv', 7)
    x_8, y_8 = hi.load_features_csv(path + '09_seawaves.csv', 8)
    x_9, y_9 = hi.load_features_csv(path + '10_thunders.csv', 9)
    x_train = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

    # load and form test data
    path = 'E:\FastDatasets\BDLib-v2\ADi\\'
    print('\nTest data from:', path)
    x_0, y_0 = hi.load_features_csv(path + '01_airplanes.csv', 0)
    x_1, y_1 = hi.load_features_csv(path + '02_alarms.csv', 1)
    x_2, y_2 = hi.load_features_csv(path + '03_applause.csv', 2)
    x_3, y_3 = hi.load_features_csv(path + '04_birds.csv', 3)
    x_4, y_4 = hi.load_features_csv(path + '05_dogs.csv', 4)
    x_5, y_5 = hi.load_features_csv(path + '06_motorcycles.csv', 5)
    x_6, y_6 = hi.load_features_csv(path + '07_rain.csv', 6)
    x_7, y_7 = hi.load_features_csv(path + '08_rivers.csv', 7)
    x_8, y_8 = hi.load_features_csv(path + '09_seawaves.csv', 8)
    x_9, y_9 = hi.load_features_csv(path + '10_thunders.csv', 9)
    x_test = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
    y_test = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

    # remove unwanted features
    r = []
    if mode == 'sti':
        for i in range(x_train.shape[1]):
            if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                r.append(i)
        r.append(0)
    elif mode == 'adi':
        for i in range(x_train.shape[1]):
            if i % 12 == 0 or i % 12 == 1 or i % 12 == 2 or i % 12 == 3 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                r.append(i)
        r.append(7)
    elif mode == 'sti-eti':
        for i in range(x_train.shape[1]):
            if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7:
                r.append(i)
        r.append(0)
    elif mode == 'adi-eti':
        for i in range(x_train.shape[1]):
            if i % 12 == 0 or i % 12 == 1 or i % 12 == 4 or i % 12 == 5:
                r.append(i)
        r.append(7)

    print('Number of initial features: {}'.format(x_train.shape[1]))
    print('Mode: {} excluding features: {}'.format(mode, r))
    x_train = np.delete(x_train, r, axis=1)
    x_test = np.delete(x_test, r, axis=1)
    print('Number of intermediate features: {}'.format(x_train.shape[1]))

    # standardize
    scale = preprocessing.StandardScaler().fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)

    #  feature selection
    # selector = GenericUnivariateSelect(f_classif, mode='fwe')  # fpr, fdr, fwe, ('k_best', param=100) mutual_info_classif f_classif
    # selector = SelectFromModel(DecisionTreeClassifier(), threshold=-np.inf, max_features=min(100, x_train_lvlib.shape[1]))
    # selector.fit(x_train_lvlib, y_train_lvlib)
    # x_train_lvlib = selector.transform(x_train_lvlib)
    # x_test_lvlib = selector.transform(x_test_lvlib)
    # print('Number of selected features: {}'.format(x_train_lvlib.shape[1]))

    # fit or predict
    score = ann.train(x_train, y_train, x_test, y_test, epochs)
    print('\nAccuracy: {:.1f}\n'.format(100*score))
    return score
def rnn_escbdlib(epochs):

    # information
    print('*****************************************************************')
    print('RNN ESC-BDLib')

    # load and form train data
    path = 'E:\FastDatasets\ESC-BDLib\STFV\\'
    print('\nTrain data from:', path)
    x_0, y_0 = hi.load_features_csv(path + '01_airplanes.csv', 0)
    x_1, y_1 = hi.load_features_csv(path + '02_alarms.csv', 1)
    x_2, y_2 = hi.load_features_csv(path + '03_applause.csv', 2)
    x_3, y_3 = hi.load_features_csv(path + '04_birds.csv', 3)
    x_4, y_4 = hi.load_features_csv(path + '05_dogs.csv', 4)
    x_5, y_5 = hi.load_features_csv(path + '06_motorcycles.csv', 5)
    x_6, y_6 = hi.load_features_csv(path + '07_rain.csv', 6)
    x_7, y_7 = hi.load_features_csv(path + '08_rivers.csv', 7)
    x_8, y_8 = hi.load_features_csv(path + '09_seawaves.csv', 8)
    x_9, y_9 = hi.load_features_csv(path + '10_thunders.csv', 9)
    x_train = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

    # load and form test data
    path = 'E:\FastDatasets\BDLib-v2\STFV\\'
    print('\nTest data from:', path)
    x_0, y_0 = hi.load_features_csv(path + '01_airplanes.csv', 0)
    x_1, y_1 = hi.load_features_csv(path + '02_alarms.csv', 1)
    x_2, y_2 = hi.load_features_csv(path + '03_applause.csv', 2)
    x_3, y_3 = hi.load_features_csv(path + '04_birds.csv', 3)
    x_4, y_4 = hi.load_features_csv(path + '05_dogs.csv', 4)
    x_5, y_5 = hi.load_features_csv(path + '06_motorcycles.csv', 5)
    x_6, y_6 = hi.load_features_csv(path + '07_rain.csv', 6)
    x_7, y_7 = hi.load_features_csv(path + '08_rivers.csv', 7)
    x_8, y_8 = hi.load_features_csv(path + '09_seawaves.csv', 8)
    x_9, y_9 = hi.load_features_csv(path + '10_thunders.csv', 9)
    x_test = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
    y_test = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

    # remove unwanted features
    r = [0]
    print('\nNumber of initial features: {}'.format(x_train.shape[1]))
    print('Excluding features: {}'.format(r))
    x_train = np.delete(x_train, r, axis=1)
    x_test = np.delete(x_test, r, axis=1)
    print('Number of intermediate features: {}'.format(x_train.shape[1]))

    # standardize
    scale = preprocessing.StandardScaler().fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)

    # reform to sequential data
    ts_length = 128
    ts_step = 64
    ts_data = np.zeros((int(x_train.shape[0] / ts_step), ts_length, x_train.shape[1]))
    ts_truth = np.zeros(int(y_train.shape[0] / ts_step))
    for i in range(0, x_train.shape[0] - ts_length, ts_step):
        ts_data[int(i / ts_step), :, :] = x_train[i:i + ts_length, :]
        ts_truth[int(i / ts_step)] = y_train[i]
    x_train = ts_data
    y_train = ts_truth
    ts_data = np.zeros((int(x_test.shape[0] / ts_step), ts_length, x_test.shape[1]))
    ts_truth = np.zeros(int(y_test.shape[0] / ts_step))
    for i in range(0, x_test.shape[0] - ts_length, ts_step):
        ts_data[int(i / ts_step), :, :] = x_test[i:i + ts_length, :]
        ts_truth[int(i / ts_step)] = y_test[i]
    x_test = ts_data
    y_test = ts_truth

    # fit or predict
    score = annlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    print('\nAccuracy: {:.1f}\n'.format(100*score))
    return score
def cnn_2d_escbdlib(epochs):

    # folding information
    print('*****************************************************************')
    print('CNN2D ESR')

    # load and form train data
    path = 'E:\FastDatasets\BDLib-v2\MEL\\'
    print('\nTrain data from:', path)
    x_0, y_0 = hi.load_spectrum_csv(path + '01_airplanes.csv', 0)
    x_1, y_1 = hi.load_spectrum_csv(path + '02_alarms.csv', 1)
    x_2, y_2 = hi.load_spectrum_csv(path + '03_applause.csv', 2)
    x_3, y_3 = hi.load_spectrum_csv(path + '04_birds.csv', 3)
    x_4, y_4 = hi.load_spectrum_csv(path + '05_dogs.csv', 4)
    x_5, y_5 = hi.load_spectrum_csv(path + '06_motorcycles.csv', 5)
    x_6, y_6 = hi.load_spectrum_csv(path + '07_rain.csv', 6)
    x_7, y_7 = hi.load_spectrum_csv(path + '08_rivers.csv', 7)
    x_8, y_8 = hi.load_spectrum_csv(path + '09_seawaves.csv', 8)
    x_9, y_9 = hi.load_spectrum_csv(path + '10_thunders.csv', 9)
    x_train = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

    # load and form test data
    path = 'E:\FastDatasets\ESC-BDLib\MEL\\'
    print('\nTest data from:', path)
    x_0, y_0 = hi.load_spectrum_csv(path + '01_airplanes.csv', 0)
    x_1, y_1 = hi.load_spectrum_csv(path + '02_alarms.csv', 1)
    x_2, y_2 = hi.load_spectrum_csv(path + '03_applause.csv', 2)
    x_3, y_3 = hi.load_spectrum_csv(path + '04_birds.csv', 3)
    x_4, y_4 = hi.load_spectrum_csv(path + '05_dogs.csv', 4)
    x_5, y_5 = hi.load_spectrum_csv(path + '06_motorcycles.csv', 5)
    x_6, y_6 = hi.load_spectrum_csv(path + '07_rain.csv', 6)
    x_7, y_7 = hi.load_spectrum_csv(path + '08_rivers.csv', 7)
    x_8, y_8 = hi.load_spectrum_csv(path + '09_seawaves.csv', 8)
    x_9, y_9 = hi.load_spectrum_csv(path + '10_thunders.csv', 9)
    x_test = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
    y_test = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

    # save here the test set (ESC-10)
    # x_train_lvlib, y_train_lvlib, x_test_lvlib, y_test
    # np.save('Temp\\bdlib_x', x_train_lvlib)
    # np.save('Temp\\bdlib_y', y_train_lvlib)
    # np.save('Temp\\esc10_x', x_test_lvlib)
    # np.save('Temp\\esc10_y', y_test)
    # return

    score = cnn2d.train(x_train, y_train, x_test, y_test, epochs)
    # score = tcnn2d.train(x_train_lvlib, y_train_lvlib, x_test_lvlib, y_test, epochs)
    print('\nAccuracy: {:.1f}\n'.format(100*score))
    return score
# transform('E:\FastDatasets\ESC-BDLib\PCM\\', 'stfv')
for i in range(2, 2):
    if i == 1: acc = np.zeros((3))
    #acc[i-1] = ann_escbdlib('all-eti', 200)
    #acc[i-1] = rnn_escbdlib(200)
    acc[i-1] = cnn_2d_escbdlib(200)
    if i == 3: print('Accuracy: {:.1f} ±{:.1f}'.format(100 * np.mean(acc), 100 * np.std(acc)))

# GTZAN Music Genre
def ann_mgr(mode, epochs, fold, folds):

    path = 'D:\\PhD\\Datasets\\GTZAN-Genre\\'
    s1 = round((folds - fold) * 1.0 / folds, 1)
    s2 = round(s1 + 1.0 / folds, 1)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load data
    x_0, y_0 = hi.load_features_csv(path + 'blues.csv', 0)
    x_1, y_1 = hi.load_features_csv(path + 'classical.csv', 1)
    x_2, y_2 = hi.load_features_csv(path + 'country.csv', 2)
    x_3, y_3 = hi.load_features_csv(path + 'disco.csv', 3)
    x_4, y_4 = hi.load_features_csv(path + 'hiphop.csv', 4)
    x_5, y_5 = hi.load_features_csv(path + 'jazz.csv', 5)
    # x_6, y_6 = hi.load_features_csv(path + 'metal.csv', 6)
    # x_7, y_7 = hi.load_features_csv(path + 'pop.csv', 7)
    # x_8, y_8 = hi.load_features_csv(path + 'raggae.csv', 8)
    # x_9, y_9 = hi.load_features_csv(path + 'rock.csv', 9)

    x_train = np.concatenate((get_array_part(x_0, 0.0, s1),
                              get_array_part(x_1, 0.0, s1),
                              get_array_part(x_2, 0.0, s1),
                              get_array_part(x_3, 0.0, s1),
                              get_array_part(x_4, 0.0, s1),
                              get_array_part(x_5, 0.0, s1),
                              # get_array_part(x_6, 0.0, s1),
                              # get_array_part(x_7, 0.0, s1),
                              # get_array_part(x_8, 0.0, s1),
                              # get_array_part(x_9, 0.0, s1),
                              get_array_part(x_0, s2, 1.0),
                              get_array_part(x_1, s2, 1.0),
                              get_array_part(x_2, s2, 1.0),
                              get_array_part(x_3, s2, 1.0),
                              get_array_part(x_4, s2, 1.0),
                              get_array_part(x_5, s2, 1.0),
                              # get_array_part(x_6, s2, 1.0),
                              # get_array_part(x_7, s2, 1.0),
                              # get_array_part(x_8, s2, 1.0),
                              # get_array_part(x_9, s2, 1.0)
                              ), axis=0)
    y_train = np.concatenate((get_array_part(y_0, 0.0, s1),
                              get_array_part(y_1, 0.0, s1),
                              get_array_part(y_2, 0.0, s1),
                              get_array_part(y_3, 0.0, s1),
                              get_array_part(y_4, 0.0, s1),
                              get_array_part(y_5, 0.0, s1),
                              # get_array_part(y_6, 0.0, s1),
                              # get_array_part(y_7, 0.0, s1),
                              # get_array_part(y_8, 0.0, s1),
                              # get_array_part(y_9, 0.0, s1),
                              get_array_part(y_0, s2, 1.0),
                              get_array_part(y_1, s2, 1.0),
                              get_array_part(y_2, s2, 1.0),
                              get_array_part(y_3, s2, 1.0),
                              get_array_part(y_4, s2, 1.0),
                              get_array_part(y_5, s2, 1.0),
                              # get_array_part(y_6, s2, 1.0),
                              # get_array_part(y_7, s2, 1.0),
                              # get_array_part(y_8, s2, 1.0),
                              # get_array_part(y_9, s2, 1.0)
                              ), axis=0)
    x_test = np.concatenate((get_array_part(x_0, s1, s2),
                              get_array_part(x_1, s1, s2),
                              get_array_part(x_2, s1, s2),
                              get_array_part(x_3, s1, s2),
                              get_array_part(x_4, s1, s2),
                              get_array_part(x_5, s1, s2),
                              # get_array_part(x_6, s1, s2),
                              # get_array_part(x_7, s1, s2),
                              # get_array_part(x_8, s1, s2),
                              # get_array_part(x_9, s1, s2)
                             ), axis=0)
    y_test = np.concatenate((get_array_part(y_0, s1, s2),
                              get_array_part(y_1, s1, s2),
                              get_array_part(y_2, s1, s2),
                              get_array_part(y_3, s1, s2),
                              get_array_part(y_4, s1, s2),
                              get_array_part(y_5, s1, s2),
                              # get_array_part(y_6, s1, s2),
                              # get_array_part(y_7, s1, s2),
                              # get_array_part(y_8, s1, s2),
                              # get_array_part(y_9, s1, s2)
                             ), axis=0)

    # remove unwanted features
    r = []
    if mode == 'sti':
        for i in range(x_train.shape[1]):
            if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                r.append(i)
        r.append(0)
    elif mode == 'adi':
        for i in range(x_train.shape[1]):
            if i % 12 == 0 or i % 12 == 1 or i % 12 == 2 or i % 12 == 3 or i % 12 == 8 or i % 12 == 9 or i % 12 == 10 or i % 12 == 11:
                r.append(i)
        r.append(3)
    elif mode == 'sti-eti':
        for i in range(x_train.shape[1]):
            if i % 12 == 4 or i % 12 == 5 or i % 12 == 6 or i % 12 == 7:
                r.append(i)
        r.append(0)
    elif mode == 'adi-eti':
        for i in range(x_train.shape[1]):
            if i % 12 == 0 or i % 12 == 1 or i % 12 == 2 or i % 12 == 3:
                r.append(i)
        r.append(3)
    else:
        #
        return 0
    print('Number of initial features: {}'.format(x_train.shape[1]))
    print('Mode: {} excluding features: {}'.format(mode, r))
    x_train = np.delete(x_train, r, axis=1)
    x_test = np.delete(x_test, r, axis=1)
    print('Number of intermediate features: {}'.format(x_train.shape[1]))

    # standardize
    scale = preprocessing.StandardScaler().fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)

    #  feature selection
    # clf = DecisionTreeClassifier()
    # trans = SelectFromModel(clf, threshold=-np.inf, max_features=max(100, x_train_lvlib.shape[1]))
    # x_train_lvlib = trans.fit_transform(x_train_lvlib, y_train_lvlib)
    # x_test_lvlib = trans.transform(x_test_lvlib)
    selector = GenericUnivariateSelect(f_classif, mode='fwe')  # fpr, fdr, fwe, ('k_best', param=125)
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)
    print('Number of selected features: {}'.format(x_train.shape[1]))

    score = ann.train(x_train, y_train, x_test, y_test, 10, epochs)
    print('Fold {}, mode {}, accuracy: {}'.format(fold, mode, round(100 * score, 1)))
    return score
for i in range(11, 11):
    if i==1: acc = np.zeros((10))
    acc[i-1] = ann_mgr('adi-eti', 200, i, 10)
    print('Mean accuracy: {:.1f} ±{:.1f}'.format(100 * np.mean(acc[:i]), 100 * np.std(acc[:i])))

# AESDD
def cnn_1d_ser_aesdd(mode, lstm, epochs, fold, folds):

    path = 'D:\\PhD\\Datasets\\AESDD-v3\\5-speakers\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('CNN1D SER')
    print(' ')
    print('Data from: ', path)
    print('Fold {}/{} with split at {:1.2f} and {:1.2f}'.format(fold, folds, s1, s2))

    # load data
    if lstm:
        x1, y1 = hi.load_audio_ts(path + 'anger.wav', 0)
        x2, y2 = hi.load_audio_ts(path + 'disgust.wav', 1)
        x3, y3 = hi.load_audio_ts(path + 'fear.wav', 2)
        x4, y4 = hi.load_audio_ts(path + 'happiness.wav', 3)
        x5, y5 = hi.load_audio_ts(path + 'sadness.wav', 4)
    else:
        x1, y1 = hi.load_audio(path + 'anger.wav', 0)
        x2, y2 = hi.load_audio(path + 'disgust.wav', 1)
        x3, y3 = hi.load_audio(path + 'fear.wav', 2)
        x4, y4 = hi.load_audio(path + 'happiness.wav', 3)
        x5, y5 = hi.load_audio(path + 'sadness.wav', 4)

    # make folds
    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x3, 0, s1),
                              get_array_part(x4, 0, s1),
                              get_array_part(x5, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),
                              get_array_part(x3, s2, 1),
                              get_array_part(x4, s2, 1),
                              get_array_part(x5, s2, 1),), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y3, 0, s1),
                              get_array_part(y4, 0, s1),
                              get_array_part(y5, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),
                              get_array_part(y3, s2, 1),
                              get_array_part(y4, s2, 1),
                              get_array_part(y5, s2, 1),), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),
                             get_array_part(x3, s1, s2),
                             get_array_part(x4, s1, s2),
                             get_array_part(x5, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),
                             get_array_part(y3, s1, s2),
                             get_array_part(y4, s1, s2),
                             get_array_part(y5, s1, s2)), axis=0)

    # run
    if lstm:
        #
        score = cnn1dlstm.train(x_train, y_train, x_test, y_test, 5, epochs)
    else:
        if mode == 'train':
            score = cnn1d.train(x_train, y_train, x_test, y_test, epochs)
        elif mode == 'retrain':
            score = cnn1d.retrain(x_train, y_train, x_test, y_test, epochs)
        elif mode == 'evaluate':
            score = cnn1d.evaluate(x_test, y_test)

    print(' ')
    print('Fold {}/{} accuracy: {:.1f}'.format(fold, folds, score * 100))
    return  score
def cnn_2d_ser_aesdd(mode, lstm, epochs, fold, folds):

    path = 'D:\\PhD\\Datasets\\AESDD-v3\\5-speakers\\MEL-22050-512-256-56\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('CNN2D SER')
    print(' ')
    print('Data from: ', path)
    print('Fold {}/{} with split at {:1.2f} and {:1.2f}'.format(fold, folds, s1, s2))

    # load data
    if lstm:
        x1, y1 = hi.load_spectrum_csv_ts(path + 'anger.csv', 0)
        x2, y2 = hi.load_spectrum_csv_ts(path + 'disgust.csv', 1)
        x3, y3 = hi.load_spectrum_csv_ts(path + 'fear.csv', 2)
        x4, y4 = hi.load_spectrum_csv_ts(path + 'happiness.csv', 3)
        x5, y5 = hi.load_spectrum_csv_ts(path + 'sadness.csv', 4)
    else:
        x1, y1 = hi.load_spectrum_csv(path + 'anger.csv', 0)
        x2, y2 = hi.load_spectrum_csv(path + 'disgust.csv', 1)
        x3, y3 = hi.load_spectrum_csv(path + 'fear.csv', 2)
        x4, y4 = hi.load_spectrum_csv(path + 'happiness.csv', 3)
        x5, y5 = hi.load_spectrum_csv(path + 'sadness.csv', 4)

    # make folds
    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x3, 0, s1),
                              get_array_part(x4, 0, s1),
                              get_array_part(x5, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),
                              get_array_part(x3, s2, 1),
                              get_array_part(x4, s2, 1),
                              get_array_part(x5, s2, 1),), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y3, 0, s1),
                              get_array_part(y4, 0, s1),
                              get_array_part(y5, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),
                              get_array_part(y3, s2, 1),
                              get_array_part(y4, s2, 1),
                              get_array_part(y5, s2, 1),), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),
                             get_array_part(x3, s1, s2),
                             get_array_part(x4, s1, s2),
                             get_array_part(x5, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),
                             get_array_part(y3, s1, s2),
                             get_array_part(y4, s1, s2),
                             get_array_part(y5, s1, s2)), axis=0)

    # run
    if lstm:
        #
        score = cnn2dlstm.train(x_train, y_train, x_test, y_test, 5, epochs)
    else:
        if mode == 'train': score = cnn2d.train(x_train, y_train, x_test, y_test, epochs)
        elif mode == 'retrain': score = cnn2d.retrain(x_train, y_train, x_test, y_test, epochs)
        elif mode == 'evaluate': score = cnn2d.evaluate(x_test, y_test)

    print(' ')
    print('Fold {}/{} accuracy: {:.1f}'.format(fold, folds, score*100))
    return score
def cnn_2d_ser_aesdd_personalized(epochs, mode):

    # information
    print(' ')
    print('CNN2D SER')

    print(' ')
    path = 'D:\\PhD\Datasets\\AESDD-v3\\6th-speaker\\train\\MEL-22050-512-256-56\\'
    print('Train data from: ', path)
    x1, y1 = hi.load_spectrum_csv(path + 'anger.csv', 0)
    x2, y2 = hi.load_spectrum_csv(path + 'disgust.csv', 1)
    x3, y3 = hi.load_spectrum_csv(path + 'fear.csv', 2)
    x4, y4 = hi.load_spectrum_csv(path + 'happiness.csv', 3)
    x5, y5 = hi.load_spectrum_csv(path + 'sadness.csv', 4)
    x_train = np.concatenate((x1, x2, x3, x4, x5), axis=0)
    y_train = np.concatenate((y1, y2, y3, y4, y5), axis=0)

    print(' ')
    path = 'D:\\PhD\Datasets\\AESDD-v3\\6th-speaker\\test\\MEL-22050-512-256-56\\'
    print('Test data from: ', path)
    x1, y1 = hi.load_spectrum_csv(path + 'anger.csv', 0)
    x2, y2 = hi.load_spectrum_csv(path + 'disgust.csv', 1)
    x3, y3 = hi.load_spectrum_csv(path + 'fear.csv', 2)
    x4, y4 = hi.load_spectrum_csv(path + 'happiness.csv', 3)
    x5, y5 = hi.load_spectrum_csv(path + 'sadness.csv', 4)
    x_test = np.concatenate((x1, x2, x3, x4, x5), axis=0)
    y_test = np.concatenate((y1, y2, y3, y4, y5), axis=0)

    if mode == 'train':
        score = cnn2d.train(x_train, y_train, x_test, y_test, epochs)
    elif mode == 'retrain':
        score = cnn2d.retrain(x_train, y_train, x_test, y_test, epochs)
    elif mode == 'evaluate':
        score = cnn2d.evaluate(x_test, y_test)
    else:
        score = -1

    print(' ')
    print('Test accuracy: {:.1f}'.format(score*100))
def cnn_1d_ser_savee(lstm, epochs, fold, folds):

    path = 'D:\\PhD\\Datasets\\SAVEE\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('CNN1D SER')
    print(' ')
    print('Data from: ', path)
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load data
    if(lstm):
        x_music, y_music = hi.load_audio_ts(path + 'Music.wav', 0)
        x_speech, y_speech = hi.load_audio_ts(path + 'Speech.wav', 1)
        x_other, y_other = hi.load_audio_ts(path + 'Others.wav', 2)
    else:
        x1, y1 = hi.load_audio(path + 'anger.wav', 0)
        x2, y2 = hi.load_audio(path + 'disgust.wav', 1)
        x3, y3 = hi.load_audio(path + 'fear.wav', 2)
        x4, y4 = hi.load_audio(path + 'happiness.wav', 3)
        x5, y5 = hi.load_audio(path + 'sadness.wav', 4)
        x6, y6 = hi.load_audio(path + 'surprise.wav', 5)

    # make folds
    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x3, 0, s1),
                              get_array_part(x4, 0, s1),
                              get_array_part(x5, 0, s1),
                              get_array_part(x6, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),
                              get_array_part(x3, s2, 1),
                              get_array_part(x4, s2, 1),
                              get_array_part(x5, s2, 1),
                              get_array_part(x6, s2, 1)), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y3, 0, s1),
                              get_array_part(y4, 0, s1),
                              get_array_part(y5, 0, s1),
                              get_array_part(y6, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),
                              get_array_part(y3, s2, 1),
                              get_array_part(y4, s2, 1),
                              get_array_part(y5, s2, 1),
                              get_array_part(y6, s2, 1)), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),
                             get_array_part(x3, s1, s2),
                             get_array_part(x4, s1, s2),
                             get_array_part(x5, s1, s2),
                             get_array_part(x6, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),
                             get_array_part(y3, s1, s2),
                             get_array_part(y4, s1, s2),
                             get_array_part(y5, s1, s2),
                             get_array_part(y6, s1, s2)), axis=0)

    # train
    if(lstm):
        #
        score = cnn1dlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    else:
        #
        score = cnn1d.train(x_train, y_train, x_test, y_test, 6, epochs)

    # print scores
    print('Fold {} with accuracy: {:.1f}'.format(fold, 100 * score))
def cnn_2d_ser_savee(lstm, epochs, fold, folds, retrain):

    path = 'D:\\PhD\\Datasets\\SAVEE\\MEL-22050-512-256-28\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('CNN2D SER')
    print(' ')
    print('Data from: ', path)
    print('Fold {}/{} with split at {:1.2f} and {:1.2f}'.format(fold, folds, s1, s2))

    # load data
    if lstm:
        x1, y1 = hi.load_spectrum_csv_ts(path + 'anger.csv', 0)
        x2, y2 = hi.load_spectrum_csv_ts(path + 'disgust.csv', 1)
        x3, y3 = hi.load_spectrum_csv_ts(path + 'fear.csv', 2)
        x4, y4 = hi.load_spectrum_csv_ts(path + 'happiness.csv', 3)
        x5, y5 = hi.load_spectrum_csv_ts(path + 'sadness.csv', 4)
    else:
        x1, y1 = hi.load_spectrum_csv(path + 'anger.csv', 0)
        x2, y2 = hi.load_spectrum_csv(path + 'disgust.csv', 1)
        x3, y3 = hi.load_spectrum_csv(path + 'fear.csv', 2)
        x4, y4 = hi.load_spectrum_csv(path + 'happiness.csv', 3)
        x5, y5 = hi.load_spectrum_csv(path + 'sadness.csv', 4)
        x6, y6 = hi.load_spectrum_csv(path + 'surprise.csv', 5)

    # make folds
    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x3, 0, s1),
                              get_array_part(x4, 0, s1),
                              get_array_part(x5, 0, s1),
                              get_array_part(x6, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),
                              get_array_part(x3, s2, 1),
                              get_array_part(x4, s2, 1),
                              get_array_part(x5, s2, 1),
                              get_array_part(x6, s2, 1)), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y3, 0, s1),
                              get_array_part(y4, 0, s1),
                              get_array_part(y5, 0, s1),
                              get_array_part(y6, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),
                              get_array_part(y3, s2, 1),
                              get_array_part(y4, s2, 1),
                              get_array_part(y5, s2, 1),
                              get_array_part(y6, s2, 1)), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),
                             get_array_part(x3, s1, s2),
                             get_array_part(x4, s1, s2),
                             get_array_part(x5, s1, s2),
                             get_array_part(x6, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),
                             get_array_part(y3, s1, s2),
                             get_array_part(y4, s1, s2),
                             get_array_part(y5, s1, s2),
                             get_array_part(y6, s1, s2)), axis=0)

    # run
    if lstm:
        #
        score = cnn2dlstm.train(x_train, y_train, x_test, y_test, 5, epochs)
    else:
        if retrain: score = cnn2d.retrain(x_train, y_train, x_test, y_test, 6, epochs)
        else:       score = cnn2d.train(x_train, y_train, x_test, y_test, 6, epochs)

    print(' ')
    print('Fold {}/{}: {}'.format(fold, folds, round(score,2)))
# transform_ser_aesdd()
for i in range(4, 4):
    # transform_ser_aesdd()
    if i == 1: acc = np.zeros((3))
    # acc[i - 1] = cnn_1d_ser_aesdd('train', False, 200, i, 3)
    acc[i - 1] = cnn_2d_ser_aesdd('train', False, 200, i, 3)
    # cnn_2d_ser_savee(False, 200, i, 10, False)
    if i == 3: print('Accuracy: {:.1f} ±{:.1f}'.format(100 * np.mean(acc), 100 * np.std(acc)))

# Trasfer Learning for SER [AESDD-v2 -> AESDD-v3]
if False:
    cnn_2d_ser_aesdd(False, 200, 1, 3, False)
    cnn_2d_ser_aesdd_personalized(200, 'evaluate')
    cnn_2d_ser_aesdd_personalized(200, 'evaluate')
    cnn_2d_ser_aesdd_personalized(200, 'evaluate')
    # cnn_2d_ser_aesdd_personalized(200, 'retrain')
    # cnn_2d_ser_aesdd_personalized(200, 'train')

# Transfer Learning for ESR [UrbanSound8k -> LVLib-SMO-v3]
if False:
    cnn_1d_esr('train', False, 200, 1, 10)
    cnn_1d_smo_lvlib('retrain', False, 200, 1, 3)
    cnn_2d_esr(False, 200, 1, 10)
    for i in range(1, 4):
        cnn_2d_smo('retrain', False, 200, i, 3)

#  MER
def cnn_2d_mer(lstm, epochs, fold, folds, retrain):

    path = 'D:\\PhD\\Datasets\\4Q\\22050-512-256-56\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('CNN2D MER')
    print(' ')
    print('Data from: ', path)
    print('Fold {}/{} with split at {:1.2f} and {:1.2f}'.format(fold, folds, s1, s2))

    # load data
    if lstm:
        x1, y1 = hi.load_spectrum_csv_ts(path + 'Q1.csv', 0)
        x2, y2 = hi.load_spectrum_csv_ts(path + 'Q2.csv', 1)
        x3, y3 = hi.load_spectrum_csv_ts(path + 'Q3.csv', 2)
        x4, y4 = hi.load_spectrum_csv_ts(path + 'Q4.csv', 3)
    else:
        x1, y1 = hi.load_spectrum_csv(path + 'Q1.csv', 0)
        x2, y2 = hi.load_spectrum_csv(path + 'Q2.csv', 1)
        x3, y3 = hi.load_spectrum_csv(path + 'Q3.csv', 2)
        x4, y4 = hi.load_spectrum_csv(path + 'Q4.csv', 3)

    # make folds
    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x3, 0, s1),
                              get_array_part(x4, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),
                              get_array_part(x3, s2, 1),
                              get_array_part(x4, s2, 1)), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y3, 0, s1),
                              get_array_part(y4, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),
                              get_array_part(y3, s2, 1),
                              get_array_part(y4, s2, 1)), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),
                             get_array_part(x3, s1, s2),
                             get_array_part(x4, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),
                             get_array_part(y3, s1, s2),
                             get_array_part(y4, s1, s2)), axis=0)

    # run
    if lstm:
        #
        score = cnn2dlstm.train(x_train, y_train, x_test, y_test, 5, epochs)
    else:
        if retrain: score = cnn2d.retrain(x_train, y_train, x_test, y_test, 4, epochs)
        else:       score = cnn2d.train(x_train, y_train, x_test, y_test, 4, epochs)

    print(' ')
    print('Fold {}/{}: {}'.format(fold, folds, round(score,2)))
for i in range(4, 4):
    # cnn_1d_emotion(epochs)
    cnn_2d_mer(False, 200, i, 3, False)

# VISUAL VAD
def cnn_2d_mouth(epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\Datasets\\M3C Speakers Localization v3\\Mouths (Diff)\\'

    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {}/{} with split at {:1.2f} and {:1.2f}'.format(fold, folds, s1, s2))
    print(' ')
    x1, y1 = hi.load_img_ts(path + '0', 0)
    x2, y2 = hi.load_img_ts(path + '1', 1)

    indices = np.random.choice(x1.shape[0], x2.shape[0], replace=False)
    x1 = x1[indices, :, :, :]
    y1 = y1[indices]

    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),), axis=0)

    # x_train_lvlib, x_test_lvlib, y_train_lvlib, y_test = train_test_split(np.concatenate((x1, x2), axis=0), np.concatenate((y1, y2), axis=0), test_size=0.33)

    # CNN LSTM evaluation
    score = cnn2dlstm.train(x_train, y_train, x_test, y_test, 2, epochs)
    print('Fold {}/{}: {:1.3f}'.format(fold, folds, round(score, 2)))

    # Simple method evaluation
    print(' ')
    x1m = np.mean(x1)
    x2m = np.mean(x2)
    # print('Full mean x1 is {:1.4f} and x2 is {:1.4f}'.format(x1m, x2m))
    correct = 0
    for i in range(0, x1.shape[0], 1):
        if np.mean(x1[i, :, :, :]) < (x1m + x2m) / 2:
            correct = correct + 1
    for i in range(0, x2.shape[0], 1):
        if np.mean(x2[i, :, :, :]) > (x1m + x2m) / 2:
            correct = correct + 1
    # print('Full accuracy is {:1.3f}'.format(correct / (x1.shape[0] + x2.shape[0])))
    x1m = 0
    x2m = 0
    c1 = 0
    c2 = 0
    for i in range(0, x_train.shape[0], 1):
        if y_train[i] == 0:
            x1m = x1m + np.mean(x_train[i, :, :, :])
            c1 = c1 + 1
        else:
            x2m = x2m + np.mean(x_train[i, :, :, :])
            c2 = c2 + 1
    x1m = x1m / c1
    x2m = x2m / c2
    # print('Test mean x1 is {:1.4f} and x2 is {:1.4f}'.format(x1m, x2m))
    correct = 0
    for i in range(0, x_test.shape[0], 1):
        if np.mean(x_test[i, :, :, :]) < (x1m + x2m) / 2 and y_test[i] == 0:
            correct = correct + 1
        elif np.mean(x_test[i, :, :, :]) > (x1m + x2m) / 2 and y_test[i] == 1:
            correct = correct + 1
    print('Test accuracy is {:1.3f}'.format(correct / (x_test.shape[0])))
    print(' ')
for i in range(4, 4):
    #
    cnn_2d_mouth(50, i, 3)