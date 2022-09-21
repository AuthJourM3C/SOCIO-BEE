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
