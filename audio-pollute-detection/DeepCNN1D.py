import time, KerasTools
import numpy as np
from scipy import stats
import tensorflow.keras as keras
from tensorflow.keras import backend, metrics, callbacks, layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Concatenate, Lambda, Subtract, UpSampling1D, ZeroPadding1D
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D#, GlobalMeanVarPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from sklearn.metrics import classification_report, confusion_matrix


class Roll1D(layers.Layer):

    def call(self, inputs):
        length = keras.backend.int_shape(inputs)[1]
        x_tile = keras.backend.tile(inputs, [1, 2, 1])

        # outputs = np.zeros((0, keras.backend.int_shape(inputs)[1], 20))
        # for i in range(0, 20):
        r = np.random.randint(1, 21)
        if np.random.randint(0, 2) > 0:
            x_roll = x_tile[:, length - r:-r, :]
        else:
            x_roll = x_tile[:, r:length + r, :]
            # outputs[:, :, i] = x_roll

        return x_roll


def train(x_train, y_train, x_test, y_test, epochs, few_shot_size=1, aggr_size=1):

    print('\nInitializing CNN1D...')
    print('{}/{} train/test samples'.format(x_train.shape[0], x_test.shape[0]))

    #  calculate classes
    if np.unique(y_train).shape[0] == np.unique(y_test).shape[0]:
        #
        num_classes = np.unique(y_train).shape[0]
    else:
        print('Error in class data...')
        return -2

    # set validation data
    '''val_size = int(0.1 * x_train.shape[0])
    r = np.random.randint(0, x_train.shape[0], size=val_size)'''
    val_size = 0.1
    length = 5
    step = int(1 / val_size * length)
    for i in range(0, x_train.shape[0] - length, step):
        if i == 0: r = []
        r.extend(range(i, i + length))
    x_val = x_train[r, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)
    print('{}/{} train/validation samples with validation parameter {}'.format(x_train.shape[0], x_val.shape[0], val_size))

    print('\nclasses:', num_classes)
    print('x_train shape:', x_train.shape), print('x_val shape:', x_val.shape), print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape), print('y_val shape:', y_val.shape), print('y_test shape:', y_test.shape)

    # shape data
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # model format
    type = 1

    # model parameters
    filters = 32
    size = 8
    stride = 1
    pool = 8

    dilation = 1
    regularizer = 0.0001
    dropout = 0.1

    # auto sequential
    if type == 0:
        model = Sequential()
        while len(model.layers) == 0:
            i = 1
            model.add(Conv1D(int(filters*i), kernel_size=max(int(size/1), 3), strides=stride, dilation_rate=dilation,
                             activation='linear', kernel_regularizer=l2(regularizer),
                             input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(MaxPooling1D(pool_size=pool))
            model.add(Dropout(dropout))
        while model.layers[len(model.layers)-1].output_shape[1]/pool > 3:
            # int(math.log(x_train_lvlib.shape[1], stride*pool))
            i = i+1
            model.add(Conv1D(int(filters*i), kernel_size=max(int(size/1), 3), strides=stride, dilation_rate=dilation, activation='elu', kernel_regularizer=l2(regularizer)))
            model.add(MaxPooling1D(pool_size=max(int(pool/1), 2)))
            model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(int(filters*i/2), activation='elu'))
        model.add(Dropout(dropout))
        model.add(Dense(num_classes, activation='softmax'))

    # lv-cnn (lvrysis originals)
    elif type == 1:
        x = Input((x_train.shape[1], x_train.shape[2]))

        m = Conv1D(filters*1, size, strides=stride, activation='relu', kernel_regularizer=l2(regularizer))(x)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters*2, size, strides=stride, activation='relu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters*3, size, strides=stride, activation='relu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters*4, size, strides=stride, activation='relu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        # m = GlobalMaxPooling1D()(m)
        m = GlobalAveragePooling1D()(m)
        # m = GlobalMeanVarPooling1D()(m)

        m = (Dense(16, activation='relu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # seqfil (sequential filterbank)
    elif type == 2:
        x = Input((x_train.shape[1], x_train.shape[2]))

        m = Conv1D(32, size, strides=size, activation='linear')(x)
        m = Dropout(dropout)(m)
        m = Conv1D(64, size, strides=size, activation='linear')(m)
        m = Dropout(dropout)(m)
        m = Conv1D(64, size, strides=size, activation='linear')(m)
        m = Dropout(dropout)(m)

        m = Conv1D(128, 5, strides=2, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = Dropout(dropout)(m)
        m = Conv1D(128, 5, strides=2, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = Dropout(dropout)(m)

        m = Flatten()(m)

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # parfil-a (parallel filterbank)
    elif type == 3:

        x = Input((x_train.shape[1], x_train.shape[2]))

        m0 = x
        m0 = Conv1D(filters, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(pool_size=512)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Flatten()(m0)
        m0 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m0)
        m0 = Dropout(dropout)(m0)

        m1 = AveragePooling1D(pool_size=2)(x)
        m1 = Conv1D(filters, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool_size=256)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Flatten()(m1)
        m1 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m1)
        m1 = Dropout(dropout)(m1)

        m2 = AveragePooling1D(pool_size=4)(x)
        m2 = Conv1D(filters, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool_size=128)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Flatten()(m2)
        m2 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m2)
        m2 = Dropout(dropout)(m2)

        m3 = AveragePooling1D(pool_size=8)(x)
        m3 = Conv1D(filters, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = MaxPooling1D(pool_size=64)(m3)
        m3 = Dropout(dropout)(m3)
        m3 = Flatten()(m3)
        m3 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m3)
        m3 = Dropout(dropout)(m3)

        m4 = AveragePooling1D(pool_size=16)(x)
        m4 = Conv1D(filters, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = MaxPooling1D(pool_size=32)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Flatten()(m4)
        m4 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m4)
        m4 = Dropout(dropout)(m4)

        m5 = AveragePooling1D(pool_size=32)(x)
        m5 = Conv1D(filters, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = MaxPooling1D(pool_size=16)(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Flatten()(m5)
        m5 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m5)
        m5 = Dropout(dropout)(m5)

        m = Concatenate(axis=1)([m0, m1, m2, m3, m4, m5])

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)

        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # parfil-b (parallel filterbank)
    elif type == 4:

        x = Input((x_train.shape[1], x_train.shape[2]))

        m0 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(x)
        m0 = MaxPooling1D(pool_size=20)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(pool_size=20)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Flatten()(m0)

        m1 = AveragePooling1D(pool_size=2)(x)
        m1 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool_size=14)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool_size=14)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Flatten()(m1)

        m2 = AveragePooling1D(pool_size=4)(x)
        m2 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool_size=10)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool_size=10)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Flatten()(m2)

        m3 = AveragePooling1D(pool_size=8)(x)
        m3 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = MaxPooling1D(pool_size=7)(m3)
        m3 = Dropout(dropout)(m3)
        m3 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = MaxPooling1D(pool_size=7)(m3)
        m3 = Dropout(dropout)(m3)
        m3 = Flatten()(m3)

        m4 = AveragePooling1D(pool_size=16)(x)
        m4 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = MaxPooling1D(pool_size=5)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = MaxPooling1D(pool_size=5)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Flatten()(m4)

        m5 = AveragePooling1D(pool_size=32)(x)
        m5 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = MaxPooling1D(pool_size=4)(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = MaxPooling1D(pool_size=4)(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Flatten()(m5)

        m = Concatenate(axis=1)([m0, m1, m2, m3, m4, m5])

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)

        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # parfil-a-dc (dedced parallel filterbank)
    elif type == 5:

        # m0 = Lambda(lambda x: x-keras.backend.mean(x))(m0)
        # m0 = keras.layers.Lambda(lambda x: keras.backend.abs(x))(m2)

        x = Input((x_train.shape[1], x_train.shape[2]))

        m0 = x
        d0 = AveragePooling1D(stride)(m0)
        d0 = UpSampling1D(stride)(d0)
        d0 = ZeroPadding1D(0)(d0)
        m0 = Subtract()([m0, d0])
        m0 = Conv1D(filters, size, strides=stride, activation='linear', kernel_regularizer=l2(regularizer))(m0)
        m0 = Dropout(dropout)(m0)
        # m0 = Subtract()([m0, d0])
        m0 = Conv1D(filters, size, activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(512)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Flatten()(m0)
        m0 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m0)
        m0 = Dropout(dropout)(m0)

        m1 = AveragePooling1D(pool_size=2)(x)
        d1 = AveragePooling1D(stride)(m1)
        d1 = UpSampling1D(stride)(d1)
        d1 = ZeroPadding1D((0, 0))(d1)
        m1 = Subtract()([m1, d1])
        m1 = Conv1D(filters, size, strides=stride, activation='linear', kernel_regularizer=l2(regularizer))(m1)
        m1 = Dropout(dropout)(m1)
        # m1 = Subtract()([m1, d1])
        m1 = Conv1D(filters, size, activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(256)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Flatten()(m1)
        m1 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m1)
        m1 = Dropout(dropout)(m1)

        m2 = AveragePooling1D(pool_size=4)(x)
        d2 = AveragePooling1D(stride)(m2)
        d2 = UpSampling1D(stride)(d2)
        d2 = ZeroPadding1D(0)(d2)
        m2 = Subtract()([m2, d2])
        m2 = Conv1D(filters, size, strides=stride, activation='linear',  kernel_regularizer=l2(regularizer))(m2)
        m2 = Dropout(dropout)(m2)
        # m2 = Subtract()([m2, d2])
        m2 = Conv1D(filters, size, activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(128)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Flatten()(m2)
        m2 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m2)
        m2 = Dropout(dropout)(m2)

        m3 = AveragePooling1D(pool_size=8)(x)
        d3 = AveragePooling1D(stride)(m3)
        d3 = UpSampling1D(stride)(d3)
        d3 = ZeroPadding1D(0)(d3)
        m3 = Subtract()([m3, d3])
        m3 = Conv1D(filters, size, strides=stride, activation='linear', kernel_regularizer=l2(regularizer))(m3)
        m3 = Dropout(dropout)(m3)
        # m3 = Subtract()([m3, d3])
        m3 = Conv1D(filters, size, activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = MaxPooling1D(64)(m3)
        m3 = Dropout(dropout)(m3)
        m3 = Flatten()(m3)
        m3 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m3)
        m3 = Dropout(dropout)(m3)

        m4 = AveragePooling1D(pool_size=16)(x)
        d4 = AveragePooling1D(stride)(m4)
        d4 = UpSampling1D(stride)(d4)
        d4 = ZeroPadding1D(1)(d4)
        m4 = Subtract()([m4, d4])
        m4 = Conv1D(filters, size, strides=stride, activation='linear', kernel_regularizer=l2(regularizer))(m4)
        m4 = Dropout(dropout)(m4)
        # m4 = Subtract()([m4, d4])
        m4 = Conv1D(filters, size, activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = MaxPooling1D(32)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Flatten()(m4)
        m4 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m4)
        m4 = Dropout(dropout)(m4)

        m5 = AveragePooling1D(pool_size=32)(x)
        d5 = AveragePooling1D(stride)(m5)
        d5 = UpSampling1D(stride)(d5)
        d5 = ZeroPadding1D((0, 1))(d5)
        m5 = Subtract()([m5, d5])
        m5 = Conv1D(filters, size, strides=stride, activation='linear', kernel_regularizer=l2(regularizer))(m5)
        m5 = Dropout(dropout)(m5)
        # m5 = Subtract()([m5, d5])
        m5 = Conv1D(filters, size, activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = MaxPooling1D(16)(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Flatten()(m5)
        m5 = (Dense(32, activation='elu', kernel_regularizer=l2(regularizer)))(m5)
        m5 = Dropout(dropout)(m5)

        m = Concatenate(axis=1)([m0, m1, m2, m3, m4, m5])

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # parfil-b-dc (dedced parallel filterbank)
    elif type == 6:

        x = Input((x_train.shape[1], x_train.shape[2]))

        m0 = x
        d0 = MaxPooling1D(int(size / 2))(m0)
        m0 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = Subtract()([m0, d0])
        m0 = MaxPooling1D(pool_size=20)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(pool_size=20)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Flatten()(m0)

        m1 = AveragePooling1D(pool_size=2)(x)
        d1 = MaxPooling1D(int(size / 2))(m1)
        m1 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = Subtract()([m1, d1])
        m1 = MaxPooling1D(pool_size=14)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool_size=14)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Flatten()(m1)

        m2 = AveragePooling1D(pool_size=4)(x)
        d2 = MaxPooling1D(int(size / 2))(m2)
        m2 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = Subtract()([m2, d2])
        m2 = MaxPooling1D(pool_size=10)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool_size=10)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Flatten()(m2)

        m3 = AveragePooling1D(pool_size=8)(x)
        d3 = MaxPooling1D(int(size / 2))(m3)
        m3 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = Subtract()([m3, d3])
        m3 = MaxPooling1D(pool_size=7)(m3)
        m3 = Dropout(dropout)(m3)
        m3 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = MaxPooling1D(pool_size=7)(m3)
        m3 = Dropout(dropout)(m3)
        m3 = Flatten()(m3)

        m4 = AveragePooling1D(pool_size=16)(x)
        d4 = MaxPooling1D(int(size / 2))(m4)
        m4 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = Subtract()([m4, d4])
        m4 = MaxPooling1D(pool_size=5)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = MaxPooling1D(pool_size=5)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Flatten()(m4)

        m5 = AveragePooling1D(pool_size=32)(x)
        d5 = MaxPooling1D(int(size / 2))(m5)
        m5 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = Subtract()([m5, d5])
        m5 = MaxPooling1D(pool_size=4)(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Conv1D(filters, size, strides=int(size / 2), activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = MaxPooling1D(pool_size=4)(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Flatten()(m5)

        m = Concatenate(axis=1)([m0, m1, m2, m3, m4, m5])

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # stepfil (stepped filterbank)
    elif type == 7:

        x = Input((x_train.shape[1], x_train.shape[2]))

        m0 = Conv1D(filters*1, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(x)
        m0 = MaxPooling1D(pool)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Conv1D(filters*2, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(pool)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Conv1D(filters*3, size, strides=int(size/2), activation='elu',kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(pool)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Flatten()(m0)

        m1 = AveragePooling1D(pool_size=2)(x)
        m1 = Conv1D(filters*1, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Conv1D(filters*2, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Conv1D(filters*3, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(int(pool/2))(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Flatten()(m1)

        m2 = AveragePooling1D(pool_size=4)(x)
        m2 = Conv1D(filters*1, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Conv1D(filters*2, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool)(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Conv1D(filters*3, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(int(pool/2))(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Flatten()(m2)

        m3 = AveragePooling1D(pool_size=8)(x)
        m3 = Conv1D(filters*1, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = MaxPooling1D(pool)(m3)
        m3 = Dropout(dropout)(m3)
        m3 = Conv1D(filters*2, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m3)
        m3 = MaxPooling1D(pool)(m3)
        m3 = Dropout(dropout)(m3)
        # m3 = Conv1D(filters*3, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m3)
        # m3 = MaxPooling1D(int(pool/2))(m3)
        # m3 = Dropout(dropout)(m3)
        m3 = Flatten()(m3)

        m4 = AveragePooling1D(pool_size=16)(x)
        m4 = Conv1D(filters*1, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = MaxPooling1D(pool)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Conv1D(filters*2, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m4)
        m4 = MaxPooling1D(pool)(m4)
        m4 = Dropout(dropout)(m4)
        m4 = Flatten()(m4)

        m5 = AveragePooling1D(pool_size=32)(x)
        m5 = Conv1D(filters*1, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = MaxPooling1D(pool)(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Conv1D(filters*2, size, strides=int(size/2), activation='elu', kernel_regularizer=l2(regularizer))(m5)
        m5 = MaxPooling1D(int(pool/2))(m5)
        m5 = Dropout(dropout)(m5)
        m5 = Flatten()(m5)

        m = Concatenate(axis=1)([m0, m1, m2, m3, m4, m5])

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # earmix (early concatenation)
    elif type == 8:

        x = Input((x_train.shape[1], x_train.shape[2]))

        c1 = Conv1D(int(filters/3), size, strides=int(stride/2), activation='relu', padding='same', kernel_regularizer=l2(regularizer))(x)
        c1 = MaxPooling1D(pool*2)(c1)
        c2 = Conv1D(int(filters/3), size, strides=stride, activation='relu', padding='same', kernel_regularizer=l2(regularizer))(x)
        c2 = MaxPooling1D(pool)(c2)
        c3 = Conv1D(int(filters/3), size, strides=stride*2, activation='relu', padding='same', kernel_regularizer=l2(regularizer))(x)
        c3 = MaxPooling1D(int(pool/2))(c3)
        #c4 = Conv1D(int(filters/4), size*8, strides=stride, activation='relu', padding='same', kernel_regularizer=l2(regularizer))(x)
        m = Concatenate(axis=2)([c1, c2, c3])
        #m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters*2, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters*3, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = GlobalAveragePooling1D()(m)

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # latmix (late concatenation)
    elif type == 9:

        x = Input((x_train.shape[1], x_train.shape[2]))

        div = 1.78

        m0 = Conv1D(int(filters * 1/div), size, strides=int(stride/2), activation='elu', kernel_regularizer=l2(regularizer))(x)
        m0 = MaxPooling1D(pool_size=pool * 2)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Conv1D(int(filters * 2/div), size, strides=int(stride/2), activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(pool_size=pool * 2)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = Conv1D(int(filters * 3/div), size, strides=int(stride/2), activation='elu', kernel_regularizer=l2(regularizer))(m0)
        m0 = MaxPooling1D(pool_size=pool * 2)(m0)
        m0 = Dropout(dropout)(m0)
        m0 = GlobalAveragePooling1D()(m0)

        m1 = Conv1D(int(filters * 1/div), size, strides=int(stride), activation='elu', kernel_regularizer=l2(regularizer))(x)
        m1 = MaxPooling1D(pool_size=pool)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Conv1D(int(filters * 2/div), size, strides=int(stride), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool_size=pool)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = Conv1D(int(filters * 3/div), size, strides=int(stride), activation='elu', kernel_regularizer=l2(regularizer))(m1)
        m1 = MaxPooling1D(pool_size=pool)(m1)
        m1 = Dropout(dropout)(m1)
        m1 = GlobalAveragePooling1D()(m1)

        m2 = Conv1D(int(filters * 1/div), size, strides=stride * 2, activation='elu', kernel_regularizer=l2(regularizer))(x)
        m2 = MaxPooling1D(pool_size=int(pool / 2))(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Conv1D(int(filters * 2/div), size, strides=stride * 2, activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool_size=int(pool / 2))(m2)
        m2 = Dropout(dropout)(m2)
        m2 = Conv1D(int(filters * 3/div), size, strides=stride * 2, activation='elu', kernel_regularizer=l2(regularizer))(m2)
        m2 = MaxPooling1D(pool_size=int(pool / 2))(m2)
        m2 = Dropout(dropout)(m2)
        m2 = GlobalAveragePooling1D()(m2)

        m = Concatenate(axis=1)([m0, m1, m2])

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # rolly
    elif type == 10:

        x = Input((x_train.shape[1], x_train.shape[2]))

        m = Concatenate(axis=2)([Roll1D()(x), Roll1D()(x), Roll1D()(x), Roll1D()(x),
                                 Roll1D()(x), Roll1D()(x), Roll1D()(x), Roll1D()(x),
                                 Roll1D()(x), Roll1D()(x), Roll1D()(x), Roll1D()(x),
                                 Roll1D()(x), Roll1D()(x), Roll1D()(x), Roll1D()(x)])
        m = Conv1D(filters * 1, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters * 2, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters * 3, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        #m = Conv1D(filters * 4, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        #m = MaxPooling1D(pool)(m)
        #m = Dropout(dropout)(m)

        m = GlobalAveragePooling1D()(m)

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # inception-like
    elif type == 11:

        x = Input((x_train.shape[1], x_train.shape[2]))

        c1 = Conv1D(int(filters/3), int(size/2), strides=stride, activation='relu', padding='same', kernel_regularizer=l2(regularizer))(x)
        c2 = Conv1D(int(filters/3), size, strides=stride, activation='relu', padding='same', kernel_regularizer=l2(regularizer))(x)
        c3 = Conv1D(int(filters/3), size*2, strides=stride, activation='relu', padding='same', kernel_regularizer=l2(regularizer))(x)
        m = Concatenate(axis=2)([c1, c2, c3])
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters * 2, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters * 3, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = GlobalAveragePooling1D()(m)

        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])

    # unet-like
    elif type == 12:
        print('unetaki')
        x = Input((x_train.shape[1], x_train.shape[2]))
        # TODO : to unetaki gia na doulepsei thelei input_shape[1] akeraio pollaplasio tou (pool_size ^ n_layers)
        # px edo einai 22050 -> 11025 -> 5512.5
        # prepei na ginei 22016 or 19968 stin kalyteri

        conv1 = Conv1D(filters, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(x)
        conv1 = Dropout(dropout)(conv1)

        pool1 = MaxPooling1D(pool_size=pool)(conv1)
        conv2 = Conv1D(filters*2, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(pool1)
        conv2 = Dropout(dropout)(conv2)

        pool2 = MaxPooling1D(pool_size=pool)(conv2)
        conv3 = Conv1D(filters*4, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(pool2)
        conv3 = Dropout(dropout)(conv3)

        pool3 = MaxPooling1D(pool_size=pool)(conv3)
        conv4 = Conv1D(filters*8, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(pool3)
        conv4 = Dropout(dropout)(conv4)
        conv4 = Conv1D(filters*8, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(conv4)

        conv3d = UpSampling1D(size=pool)(conv4)
        conv3d = Conv1D(filters*4, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(conv3d)
        conv3d = Dropout(dropout)(conv3d)
        merge3 = Concatenate(axis=-1)([conv3, conv3d])

        conv2d = UpSampling1D(size=pool)(merge3)
        conv2d = Conv1D(filters*2, size, activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(conv2d)
        conv2d = Dropout(dropout)(conv2d)
        merge2 = Concatenate(axis=-1)([conv2, conv2d])

        conv1d = UpSampling1D(size=pool)(merge2)
        conv1d = Conv1D(filters, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(conv1d)
        conv1d = Dropout(dropout)(conv1d)
        merge1 = Concatenate(axis=-1)([conv1, conv1d])

        conv0d = Conv1D(2, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(merge1)
        conv0d = Conv1D(1, size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(regularizer))(conv0d)
        conv0d = Dropout(dropout)(conv0d)

        x = Flatten()(conv0d)
        x = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(x)
        x = Dropout(dropout)(x)
        y = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=[x], outputs=[y])

    # sincnet
    elif type == 13:
        x = Input((x_train.shape[1], x_train.shape[2]))

        m = KerasTools.SincConv1D(filters, 251, 22050)(x)
        m = MaxPooling1D(pool)(m)
        m = KerasTools.LayerNormalization()(m)
        m = LeakyReLU()(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters * 2, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = Conv1D(filters * 3, size, strides=stride, activation='elu', kernel_regularizer=l2(regularizer))(m)
        m = MaxPooling1D(pool)(m)
        m = Dropout(dropout)(m)

        m = GlobalAveragePooling1D()(m)
        m = (Dense(64, activation='elu', kernel_regularizer=l2(regularizer)))(m)
        m = Dropout(dropout)(m)
        y = Dense(num_classes, activation='softmax')(m)

        model = Model(inputs=[x], outputs=[y])


    # summarize model
    for i in range(0, len(model.layers)):
        # if i == 0:
            # keras.utils.plot_model(model, to_file='Models\\model_cnn1d.png')
            # f = open('Models\\model_cnn1d.txt', 'w')
            # print(' ')
        # try:
        #     print('{}. Layer {} with kernel and I/O shapes: {} | {} | {}'.format(i, model.layers[i].name, model.layers[i].kernel.shape, model.layers[i].input_shape, model.layers[i].output_shape))
        # except:
        #     print('{}. Layer {} with kernel and I/O shapes: {} | {} | {}'.format(i, model.layers[i].name, model.layers[i].input_shape, "n/a", model.layers[i].output_shape))
        # f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        if i == len(model.layers) - 1:
            # f.close()
            print(' ')
            model.summary()

    # compile, fit, evaluate
    callback = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, restore_best_weights=True)]
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=epochs, verbose=2, validation_data=(x_val, y_val), callbacks=callback)
    score = model.evaluate(x_test, y_test, verbose=2)

    # present results
    y_pred = model.predict(x_test)
    # print('\n', confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    print('\n', classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), output_dict=False))

    # evaluate on larger frames
    for i in range(0, y_test.shape[0] - aggr_size, aggr_size):
        if i == 0:
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)
            y_aggr_test = []
            y_aggr_pred = []
        if np.unique(y_test[i:i + aggr_size]).shape[0] == 1:
            y_aggr_test.append(stats.mode(y_test[i:i + aggr_size])[0][0])
            y_aggr_pred.append(stats.mode(y_pred[i:i + aggr_size])[0][0])
    scipy_score = classification_report(y_aggr_test, y_aggr_pred, output_dict=True)['accuracy']
    print('Short {:.2f} and aggregated {:.2f}\n'.format(score[1], scipy_score))

    # save model
    model.save('Models\\model_cnn1d_{}class.h5'.format(num_classes))

    # return results
    return score[1]


def retrain(x_train, y_train, x_test, y_test, epochs, batch_size=128, verbose=2):

    #  calculate classes
    if np.unique(y_train).shape[0] == np.unique(y_test).shape[0]:
        #
        num_classes = np.unique(y_train).shape[0]
    else:
        print('Error in class data...')
        return -2

    # set validation data
    val_size = int(0.1 * x_train.shape[0])
    r = np.random.randint(0, x_train.shape[0], size=val_size)
    x_val = x_train[r, :, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)

    print(' ')
    print('Retrieving CNN1D...')
    print('classes:', num_classes)
    print('x train shape:', x_train.shape), print('x val shape:', x_val.shape), print('x test shape:', x_test.shape)
    print('y train shape:', y_train.shape), print('y val shape:', y_val.shape), print('y test shape:', y_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # load model
    json_file = open('Models\\model_cnn1d.json', 'r')
    model = model_from_json(json_file.read())
    json_file.close()
    model.load_weights("Models\\model_cnn1d.h5")

    # freeze dense layers and reshape the last
    regularizer = 0.0025
    dropout = 0.25
    for i in range(0, len(model.layers)):
        #
        model.layers[i].trainable = False
    tl_model = Sequential()
    tl_model.add(Model(model.input, model.layers[-3].output))
    tl_model.add(Dense(64, activation='elu', kernel_regularizer=l2(regularizer), name='new_dense_1'))
    tl_model.add(Dropout(dropout, name='new_dropout_1'))
    tl_model.add(Dense(num_classes, activation='softmax', name='new_dense_2'))

    # summarize model
    for i in range(0, len(tl_model.layers)):
        # if i == 0:
            # print(' ')
        # print('{}. Layer {} with input / output shapes: {} / {}'.format(i, tl_model.layers[i].name, tl_model.layers[i].input_shape, tl_model.layers[i].output_shape))
        if i == len(tl_model.layers) - 1:
            print(' ')
            tl_model.summary()

    # compile, fit evaluate
    callback = [callbacks.EarlyStopping(monitor='val_acc', min_delta=0.005, patience=20, restore_best_weights=True)]
    tl_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    tl_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_val, y_val), callbacks=callback)
    time.sleep(1)
    score = tl_model.evaluate(x_test, y_test, verbose=2)
    time.sleep(2)

    # results
    return score[1]