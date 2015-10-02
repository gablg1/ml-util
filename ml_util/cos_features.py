import numpy as np

class Generator:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.A, self.b = self.createFeatureMatrix()

    def createFeatureMatrix(self):
        A = np.random.multivariate_normal(np.zeros(self.in_dim),
                                          np.identity(self.in_dim),
                                          size=self.out_dim)
        b = np.random.uniform(0, 2 * np.pi, size=self.out_dim)
        assert(A.shape == (self.out_dim, self.in_dim))
        assert(b.shape == (self.out_dim,))
        return (A, b)

    def dataToFeatures(self, X):
        assert(X.shape[1] == self.in_dim)
        assert(X.shape[1] == self.A.T.shape[0])
        features = X.dot(self.A.T)

        # We replicate an instance of b for each data point
        num_data_points = features.shape[0]
        B = np.tile(self.b, (num_data_points, 1))
        return np.cos(features + B)
