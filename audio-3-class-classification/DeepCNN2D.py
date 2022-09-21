import time
import numpy as np
from scipy import stats
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend, metrics, callbacks, regularizers
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GaussianNoise, GaussianDropout, Lambda, Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import plot_model, multi_gpu_model, to_categorical


def train(x_train, y_train, x_test, y_test, epochs, few_shot_size=1, aggr_size=1):

    print('\nInitializing CNN2D...')
    print('{}/{} train/test samples'.format(x_train.shape[0], x_test.shape[0]))

    #  calculate classes
    if np.unique(y_train).shape[0] == np.unique(y_test).shape[0]:
        #
        num_classes = np.unique(y_train).shape[0]
    else:
        print('Error in class data...')
        return -2

    # set validation data
    '''try:
        #
        r = np.load('Temp/rand.npy')
    except:
        r = np.random.randint(0, x_train_lvlib.shape[0], size=int(0.1 * x_train_lvlib.shape[0]))
        np.save('Temp/rand.npy', r)
    x_val = x_train_lvlib[r, :, :]
    y_val = y_train_lvlib[r]
    x_train_lvlib = np.delete(x_train_lvlib, r, axis=0)
    y_train_lvlib = np.delete(y_train_lvlib, r, axis=0)'''
    val_size = 0.1
    length = 5
    step = int(1/val_size*length)
    for i in range(0, x_train.shape[0]-length, step):
        if i == 0: r = []
        r.extend(range(i, i+length))
    x_val = x_train[r, :, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)
    print('{}/{} train/validation samples with validation parameter {}'.format(x_train.shape[0], x_val.shape[0], val_size))

    # perform few shot trim
    '''r = np.random.choice(x_train_lvlib.shape[0], size=int((1 - few_shot_size) * x_train_lvlib.shape[0]), replace=False)
    size = int((1 - few_shot_size) * x_train_lvlib.shape[0])'''
    if few_shot_size > 1:   step = int(x_train.shape[0] / few_shot_size)
    else:                   step = int(1/few_shot_size)
    for i in range(0, x_train.shape[0], step):
        if i == 0: r = []
        r.append(i)
    x_train = x_train[r, :, :]
    y_train = y_train[r]
    print('{} train samples with few shot parameter {}'.format(x_train.shape[0], few_shot_size))

    print('\nclasses:', num_classes)
    print('x train shape:', x_train.shape), print('x val shape:', x_val.shape), print('x test shape:', x_test.shape)
    print('y train shape:', y_train.shape), print('y val shape:', y_val.shape), print('y test shape:', y_test.shape)

    # shape data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # define the model
    activation = 'relu'
    dropout = 0.1

    # preprocessing
    '''
    offset = 1.0 * np.std(x_train_lvlib)
    dc0 = (x)
    dc1 = GaussianNoise(offset*0.1)(x)
    dc2 = GaussianDropout(dropout)(x)
    dc3 = Lambda(lambda r: r + __import__('keras').backend.random_uniform((1,), -offset, offset))(x)
    dc4 = Lambda(lambda r: r + __import__('keras').backend.random_uniform((1,), -offset, offset))(x)
    m = Concatenate()([dc0, dc1, dc2, dc3, dc4])
    m = Lambda(lambda r: r - __import__('keras').backend.mean(r))(x)
    '''

    # sequential
    '''model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation=activation, input_shape=(x_train_lvlib.shape[1], x_train_lvlib.shape[2], 1)))
    model.add(LayerNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(32, kernel_size=3, activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, kernel_size=3, activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(128, kernel_size=3, activation=activation))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))'''

    # functional
    x = Input((x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    m = Conv2D(16, 3)(x)
    m = Activation(activation)(m)
    m = MaxPooling2D((2, 2))(m)
    m = Dropout(dropout)(m)

    m = Conv2D(32, 3)(m)
    m = Activation(activation)(m)
    m = MaxPooling2D((2, 2))(m)
    m = Dropout(dropout)(m)

    m = Conv2D(64, 3)(m)
    m = Activation(activation)(m)
    if x_train.shape[1] > 40:   m = MaxPooling2D((2, 2))(m)
    else:                       m = GlobalAveragePooling2D()(m)
    m = Dropout(dropout)(m)

    if x_train.shape[1] > 40:
        m = Conv2D(128, 3)(m)
        m = Activation(activation)(m)
        m = GlobalAveragePooling2D()(m)
        m = Dropout(dropout)(m)

    m = (Dense(64))(m)
    m = Activation(activation)(m)
    m = Dropout(dropout)(m)

    m = (Dense(32))(m)
    m = Activation(activation)(m)
    m = Dropout(dropout)(m)

    y = Dense(num_classes, activation='softmax')(m)

    model = Model(inputs=[x], outputs=[y])

    # summarize model
    for i in range(0, len(model.layers)):
        #if i == 0:
            #plot_model(model, to_file='Models\\model_cnn2d.png')
            # f = open('Models\\model_cnn2d.txt', 'w')
            # print(' ')
        # print('{}. Layer {} with input / output shapes: {} / {}'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
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
    print('Short {:.2f} and aggregated {:.2f}'.format(score[1], scipy_score))

    # save model
    model.save('Models\\model_cnn2d_{}class.h5'.format(num_classes))

    # results
    if aggr_size > 1:   return scipy_score
    else:               return score[1]


def retrain(x_train, y_train, x_test, y_test, epochs, batch_size=256, verbose=2):

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
    print('Retrieving CNN2D...')
    print('classes:', num_classes)
    print('x train shape:', x_train.shape), print('x val shape:', x_val.shape), print('x test shape:', x_test.shape)
    print('y train shape:', y_train.shape), print('y val shape:', y_val.shape), print('y test shape:', y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # load model and weights
    # model = model_from_json(json_file.read(open('Models\\model_cnn2d.json', 'r')))
    model = Model.from_config(pickle.load(open('Models\\model_cnn2d.pickle', 'rb')))
    model.load_weights("Models\\model_cnn2d.h5")

    # freeze dense layers and reshape the last
    regularizer = 0.0025
    dropout = 0.25
    for i in range(0, len(model.layers)):
        #
        model.layers[i].trainable = False
    tl_model = Sequential()
    tl_model.add(Model(model.input, model.layers[-1].output))
    # tl_model.add(Dense(64, activation='elu', kernel_regularizer=regularizers.l2(regularizer), name='new_dense_1'))
    # tl_model.add(Dropout(dropout, name='new_dropout_1'))
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
    callback = [callbacks.EarlyStopping(monitor='val_acc', patience=20, restore_best_weights=True)]
    tl_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    tl_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_val, y_val), callbacks=callback)
    time.sleep(1)
    score = tl_model.evaluate(x_test, y_test, verbose=2)
    time.sleep(2)

    # results
    return score[1]


def evaluate(x_test, y_test):

    #  calculate classes
    num_classes = np.unique(y_test).shape[0]

    print(' ')
    print('Retrieving CNN2D...')
    print('classes:', num_classes)
    print('x test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # load model and weights
    model = model_from_json(open('Models\\model_cnn2d.json', 'r').read())

    # compile and evaluate
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=2)
    time.sleep(5)

    # results
    return score[1]


def predict(x_test):

    x_test = x_test[:1, :, :]
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    model = model_from_json(open('Models\\model_cnn2d.json', 'r').read())

    times = []
    for i in range(100):
        t1 = time.time_ns()
        model.predict(x_test, batch_size=1, verbose=0)
        t2 = time.time_ns()
        print('{:.2f}'.format((t2 - t1) / 1000000))
        times.append((t2 - t1) / 1000000)
        time.sleep(0.3)
        k = np.random.normal(1, 1, 84)
        for i in range(1000):
            a = np.mean(k)
            a = np.std(k)
            a = stats.skew(k)
            a = stats.kurtosis(k)
            # fit_levy(k)
    print('avg', np.mean(times))


    k = np.random.normal(1, 1, 84)
    t1 = time.time_ns()
    for i in range(25):
        a = np.mean(k)
        a = np.std(k)
        a = stats.kurtosis(k)
        a = stats.skew(k)
        # fit_levy(k)
    t2 = time.time_ns()
    print('{:.2f}'.format((t2 - t1) / 1000000))

    t1 = time.time_ns()
    for i in range(1):
        model.predict(x_test, batch_size=1, verbose=0)
    t2 = time.time_ns()
    print('{:.2f}'.format((t2 - t1) / 1000000))