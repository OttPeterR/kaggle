import numpy as np

def get_train_data():
    data = np.genfromtxt('train.csv', delimiter=',')
    return data[1:] # data[0] is the labels

if __name__== "__main__":
    print("digit-recognizer")