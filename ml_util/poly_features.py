import numpy as np
import ml

class Generator:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def datumToFeatures(self, x):
        return np.array([x ** i for i in range(1, self.out_dim + 1)])

    def dataToFeatures(self, X):
        assert(ml.dim(X) == 1)

        features = np.array([self.datumToFeatures(x) for x in X])

        assert(features.shape[1] == self.out_dim)
        assert(features.shape[0] == X.shape[0])
        return features
