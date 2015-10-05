import numpy as np
import ml

class Generator:
    def __init__(self, out_dim, include_constant=True):
        self.out_dim = out_dim
        self.include_constant = include_constant

    def datumToFeatures(self, x):
        assert(ml.isScalar(x) or ml.isVector(x))
        start = 0
        end = self.out_dim
        if not self.include_constant:
        	start += 1
        	end += 1
        return np.array([x ** i for i in range(start, end)])

    def dataToFeatures(self, X):
        assert(ml.dim(X) == 1)

        features = np.array([self.datumToFeatures(x) for x in X])

        assert(features.shape[1] == self.out_dim)
        assert(features.shape[0] == X.shape[0])
        return features
