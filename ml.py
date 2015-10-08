import numpy as np
import math

#### Numpy related helpers

# Whether the given v is a one dimensional vector
def isVector(v):
    if isScalar(v):
        return False
    return len(v.shape) == 1

def isScalar(s):
    return np.isscalar(s)

def dim(m):
    return len(m.shape)

def canDot(a, b):
    return dim(a) == 1 and dim(b) == 1 and a.shape == b.shape

def canMultiply(a, b):
    if dim(a) == 2 and dim(b) in [1, 2]:
        ret = a.shape[1] == b.shape[0]
    elif dim(a) == 1 and dim(b) in [1, 2]:
        ret = a.shape[0] == b.shape[0]
    else:
        ret = False

    if not ret:
        print "Shape mismatch", a.shape, b.shape
    return ret

def canSum(a, b):
    if a.shape != b.shape:
        print "Shape mismatch", a.shape, b.shape
        return False
    return True


### Data related helpers

# Splits the data between Train and Test
def splitTrainTest(X, y, fraction_train = 9.0 / 10.0):
    end_train = round(X.shape[ 0 ] * fraction_train)
    X_train = X[0 : end_train, ]
    y_train = y[ 0 : end_train ]
    X_test = X[ end_train :, ]
    y_test = y[ end_train : ]
    return X_train, y_train, X_test, y_test

# Normalizes the features so they have mean 0. and stdev 1.
def normalizeFeatures(X_train, X_test):
    mean_X_train = np.mean(X_train, 0)
    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    X_train_normalized = (X_train - mean_X_train) / std_X_train
    X_test_normalized = (X_test - mean_X_train) / std_X_train
    return X_train_normalized, X_test_normalized


# Calculates the RMSE between the two (1, N) matrices
def rmse(predictions, targets):
    assert(canSum(predictions, targets))
    return np.sqrt(np.mean(np.square(predictions-targets)))


def predictRMSE(f, x, y, kind=None):
    # Predictions of the train data
    vf = np.vectorize(f)
    y_predictions = vf(x)
    data_rmse = rmse(y_predictions, y)

    if kind:
        print "RMSE on %s data: %s" % (kind, data_rmse)
    else:
        print "RMSE on data: %s" % data_rmse
    print
    return data_rmse

def testAndTrainRMSE(f, X_test, y_test, X_train, y_train):
    # Predictions of the train data
    train_rmse = predictRMSE(f, X_train, y_train, kind='train')

    # Predictions of the test data
    test_rmse = predictRMSE(f, X_test, y_test, kind='test')
    return test_rmse, train_rmse



## Gradient Stuff

# Returns the negative log of the given gaussian PDF
# evaluated at value
#
# The result is correct up to an additive constant
def applyNLGaussian(value, mean, var):
    assert(value.shape == mean.shape)
    term = value - mean
    assert(isvector(term))
    return (1. / var) * term.dot(term.T)

# Calculates the derivative of f on x along axis i
def axisDerivative(f, x, i, D, epsilon=0.001):
    assert(x.shape == (D,))
    epsilon_matrix = np.zeros(D)
    epsilon_matrix[i] += epsilon

    ret = (f(x + epsilon_matrix) - f(x - epsilon_matrix)) / (2 * epsilon)
    assert(np.isscalar(ret))
    return ret

# f is a mapping (D,) -> 1 and grad is a mapping (D,) -> (D,)
def testGradient(f, grad, D):
    test = np.random.rand(D)
    assert(test.shape == (D,))
    calculated_gradient = grad(test)
    numeric_gradient = []
    for i in range(D):
        numeric_gradient.append(axisDerivative(f, test, i, D))
    numeric_gradient = np.array(numeric_gradient)

    print 'The numeric gradient is:'
    print numeric_gradient.T
    print
    print 'The solved gradient is:'
    print calculated_gradient.T
    print

    grad_rmse = rmse(numeric_gradient, calculated_gradient)
    print 'The RMSE between them is %f' % grad_rmse
    print

## Linear regression

# Solves linear regression using the numerically stable QR method
# (N x D, D) -> (D)
def QRRegression(X_train, y_train):
    # If the number of data points is less than the dimension,
    # the Linear regression solution is not well defined
    assert(X_train.shape[0] >= X_train.shape[1])

    Q, R = np.linalg.qr(X_train)
    assert(R.shape[0] == R.shape[1])
    R_inv = np.linalg.inv(R)
    assert(Q.shape[0] == y_train.shape[0])
    return R_inv.dot(Q.T).dot(y_train)

# This wasn't tested for dim(X_train) != 2 or dim(Y_train) != 1
def ridgeData(X_train, Y_train, D, ridge_var):
    ridge_precision = 1. / ridge_var
    ridge_matrix = np.sqrt(ridge_precision) * np.identity(D)
    X_trainp = np.concatenate((X_train, ridge_matrix), 0)

    zeros = np.zeros([D for i in range(dim(Y_train))])
    assert(dim(zeros) == dim(Y_train))
    Y_trainp = np.concatenate((Y_train, zeros), 0)
    return X_trainp, Y_trainp

def linRegTestAndTrainRMSE(beta, X_test, y_test, X_train, y_train):
    assert(isvector(beta))
    return testAndTrainRMSE(lambda x: x.dot(beta), X_test, y_test, X_train, y_train)


## splits a matrix of observations into k different folds, and returns
# another matrix of k-1 folds and 1 additional fold separated out
# x is the matrix of observations
# index is which of the 0...k-1 folds one wants separated out
# folds is the total number of partitions desired

def kfold(x, index, folds=10):
    if (index > folds-1) or (index < 0):
        raise IndexError('Index out of range of permitted folds')

    if folds < 2:
        raise ValueError('Insufficient number of folds')

    observations = x.shape[0]

    if observations < folds:
        raise IndexError('Cannot have more folds than observations')

    indices = [(observations/folds)*i for i in xrange(1,10)]
    splits = np.array_split(x, indices)
    test = splits.pop(index)

    return np.concatenate(splits), test

## Priors and Posteriors and Likelihoods

# Model assumes Y ~ N(Xw,Sigma)
# and that w ~ N(w_0, V_0) is the prior
#
# (D, D X D, N X D, N, N X N) -> (D, D X D)
# Returns the posterions w_n, V_n given the data X, Y
def bayesLinRegPosterior(w_0, V_0, X, Y, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    V_0_inv = np.linalg.inv(V_0)

    # After this point we know all invertible matrices are square
    assert(canMultiply(X.T, Sigma_inv))
    V_n = np.linalg.inv(V_0_inv + X.T.dot(Sigma_inv).dot(X))

    assert(canMultiply(X.T, Y))
    assert(canMultiply(V_0_inv, w_0))
    w_n = V_n.dot(V_0_inv.dot(w_0) + X.T.dot(Sigma_inv).dot(Y))
    return w_n, V_n

## Logistic Regression

def sigmoid(x):
    assert(isScalar(x))
    ret = 1 / (1 + np.exp(-x))
    assert(0 <= ret and ret <= 1)
    return ret
