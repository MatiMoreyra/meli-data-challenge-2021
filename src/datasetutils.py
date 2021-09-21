import numpy as np

import config

def load_lvl_0_dataset(repetitions=1, features=None):
    x_train_list = list()
    x_test_list = list()
    y_train_list = list()
    y_test_list = list()
    t_train_list = list()
    t_test_list = list()

    for r in range(repetitions):
        data = np.load(config.SL_DATASET_TEMPLATE.format(r))
        X = (data["S"])
        Y = data["Y"]
        T = data["T"]
        if (features == None):
            features = range(1, X.shape[2])
        x_train_list.append(X[:config.LVL_0_TRAIN_SIZE, :, features])
        y_train_list.append(Y[:config.LVL_0_TRAIN_SIZE])
        t_train_list.append(T[:config.LVL_0_TRAIN_SIZE])
        x_test_list.append(X[config.LVL_0_TRAIN_SIZE:, :, features])
        y_test_list.append(Y[config.LVL_0_TRAIN_SIZE:])
        t_test_list.append(T[config.LVL_0_TRAIN_SIZE:])

    X_train = np.vstack(x_train_list)
    Y_train = np.vstack(y_train_list)
    T_train = np.vstack(t_train_list)

    X_test = np.vstack(x_test_list)
    Y_test = np.vstack(y_test_list)
    T_test = np.vstack(t_test_list)

    return X_train, T_train, Y_train, X_test, T_test, Y_test

def load_lvl_1_dataset(repetitions=1, features=None):
    x_train_list = list()
    x_test_list = list()
    y_train_list = list()
    y_test_list = list()
    t_train_list = list()
    t_test_list = list()

    for r in range(repetitions):
        data = np.load(config.SL_DATASET_TEMPLATE.format(r))
        X = (data["S"])[config.LVL_0_TRAIN_SIZE:]
        Y = data["Y"][config.LVL_0_TRAIN_SIZE:]
        T = data["T"][config.LVL_0_TRAIN_SIZE:]
        if (features == None):
            features = range(1, X.shape[2])
        x_train_list.append(X[:config.LVL_1_TRAIN_SIZE, :, features])
        y_train_list.append(Y[:config.LVL_1_TRAIN_SIZE])
        t_train_list.append(T[:config.LVL_1_TRAIN_SIZE])
        x_test_list.append(X[config.LVL_1_TRAIN_SIZE:, :, features])
        y_test_list.append(Y[config.LVL_1_TRAIN_SIZE:])
        t_test_list.append(T[config.LVL_1_TRAIN_SIZE:])

    X_train = np.vstack(x_train_list)
    Y_train = np.vstack(y_train_list)
    T_train = np.vstack(t_train_list)

    X_test = np.vstack(x_test_list)
    Y_test = np.vstack(y_test_list)
    T_test = np.vstack(t_test_list)

    return X_train, T_train, Y_train, X_test, T_test, Y_test
