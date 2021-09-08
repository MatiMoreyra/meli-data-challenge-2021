import numpy as np
import pickle


class InputScaler():
    def __init__(self):
        self.X_means = list()
        self.X_stds = list()
        self.T_mean = 0
        self.T_std = 0

    def fit(self, X, T):
        for f in range(X.shape[2]):
            self.X_means.append(np.mean(X[:, :, f]))
            self.X_stds.append(np.std(X[:, :, f]))
        self.T_mean = np.mean(T)
        self.T_std = np.std(T)

    def transform(self, X, T):
        for f in range(X.shape[2]):
            X[:, :, f] = (X[:, :, f] - self.X_means[f]) / \
                np.clip(self.X_stds[f], 1e-6, None)
        T = (T - self.T_mean) / np.clip(self.T_std, 1e-6, None)
        return X, T

    def dump(self, path):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))
