import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential


def get_data(file, delimiter=','):
    data = np.genfromtxt(file, delimiter=delimiter)
    return data


def get_train_data():
    train_data = get_data('train.csv')[1:]
    train_X = train_data[:, 3:]
    train_Y = train_data[:, 1:3]
    return train_X, train_Y


def get_test_data():
    # 4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
    # data is not guarenteed to have all fields
    return get_data('test.csv')[1:]

def filter_data(x):




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
               header='PassengerId,Survived')


def train_model(X, Y):

    model = Sequential()
    model.add(Dense(450, input_dim=784, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='sigmoid'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, epochs=500, batch_size=4200)
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
        prepare_date_for_submission(predictions)
        # np.set_printoptions(threshold=np.nan)
        print('Complete')

