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
