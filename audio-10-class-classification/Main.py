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
import DeepCNN1D as cnn1d, DeepCNN2D as cnn2d


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
