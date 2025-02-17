import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.optimizers import SGD



def get_data(file, delimiter=','):
    data = np.genfromtxt(file, delimiter=delimiter)
    return data


def get_train_data():
    train_data = get_data('train.csv')[1:]
    train_X = train_data[:, 1:]
    train_Y = train_data[:, :1]
    train_Y = expand_train_data(train_Y)
    return train_X, train_Y


def get_test_data():
    return get_data('test.csv')[1:]


def expand_train_data(y):
    # expands 0 through 9 values into rows
    # so a [3] will turn to a [0,0,0,1,0,0,0,0,0,0]
    new_Y = np.zeros([y.shape[0], 10])
    for ind in range(y.shape[0]):
        new_Y[ind][int(y[ind][0])] = 1
        ind += 1
    return new_Y


def reduce_prediction_data(y):
    # reduces [0.1,0,0.1,0.7,0,0,0.1,0,0,0] into a 3
    return y.argmax(1)


def prepare_date_for_submission(y, filename='submission.csv'):
    # takes y as [value0, value1, value2, ...]
    # outputs as: (yes, the # is included in the first line)
    # # ImageId,Label
    # 1,value0
    # 2,value1
    # 3,value2
    sub_data = np.zeros([y.shape[0], 2])
    count = 0
    for val in y:
        sub_data[count] = [count + 1, val]
        count += 1
    sub_data = sub_data.astype(int)
    np.savetxt(fname=filename,
               X=sub_data,
               fmt='%i',
               delimiter=',',
               comments='',
               header='ImageId,Label')


def train_model(X, Y):

    input_shape = (28, 28, 1)

    model = Sequential()
    model.add(Reshape(input_shape, input_shape=(X.shape[1],)))

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.33))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.33))


    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    num_batches = 10
    batch_size = int(X.shape[0]/num_batches)
    epochs = 150
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    return model


def test_model(model, X, Y):
    return model.evaluate(X, Y)


def predict(model, X):
    return model.predict(X)


if __name__ == '__main__':
    np.random.seed(0)
    
    # Use GPU?
    if True:
        config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
        sess = tf.Session(config=config) 
        keras.backend.set_session(sess)


    # training
    train_X, train_Y = get_train_data()
    model = train_model(train_X, train_Y)

    # testing
    scores = test_model(model, train_X, train_Y)
    print('Scores:')
    print(scores)

    # predicting
    run_prediction = True
    if run_prediction:
        test_data = get_test_data()
        predictions = predict(model, test_data)
        predictions = reduce_prediction_data(predictions)
        prepare_date_for_submission(predictions)
        # np.set_printoptions(threshold=np.nan)
        print('Complete')
