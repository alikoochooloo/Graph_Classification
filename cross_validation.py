from sklearn.metrics import confusion_matrix
import numpy as np


def cross_val(k, x_train, y):
    fold_len = len(x_train) // k

    # Partition data into k-folds
    D, Y = [], []
    for i in range(0, len(x_train), fold_len):
        D.append(x_train[i: i + fold_len])
        Y.append(y[i: i + fold_len])
    D = D[:-1]  # Got ride of the array of size 2 because bugs
    Y = Y[:-1]

    # Goes through every subset
    xtrain_all, xtest_all, ytrain_all, ytest_all = [], [], [], []
    for i in range(len(D)):

        # This gets the current training and testing folds
        test, y_test = D.pop(i), Y.pop(i)
        train, y_train = np.array([j for i in D for j in i]), np.array([j for i in Y for j in i])
        D.insert(i, test)
        Y.insert(i, y_test)

        # Add the data to the lists to return
        xtrain_all.append(train)
        xtest_all.append(test)
        ytrain_all.append(y_train)
        ytest_all.append(y_test)

    return xtrain_all, xtest_all, ytrain_all, ytest_all