import numpy as np
from keras.layers import Dense
from keras.models import Sequential


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


def sigmoid(val):
    return 1.0 / (1.0 + (np.exp(-val)))


def train_model(X, Y):

    model = Sequential()
    model.add(Dense(12, input_dim=784, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, epochs=30, batch_size=4)
    return model


def test_model(model, X, Y):
    return model.evaluate(X, Y)


def predict(model, X):
    return model.predict(X)


if __name__ == '__main__':
    np.random.seed(0)

    # training
    train_X, train_Y = get_train_data()
    model = train_model(train_X, train_Y)

    # testing
    scores = test_model(model, train_X, train_Y)
    print('Scores:')
    print(scores)

    # predicting
    run_prediction = False
    if run_prediction:
        test_data = get_test_data()
        predictions = predict(model, train_X)
        predictions = reduce_prediction_data(predictions)
        np.set_printoptions(threshold=np.nan)
        print("Predictions:    " + str(predictions))
        print("Correct Values: " + str(train_Y))
