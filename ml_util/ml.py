import numpy as np

#### Numpy related helpers

# Whether the given v is a one dimensional vector
def isvector(v):
    if np.isscalar(v):
    	return False
    return len(v.shape) == 1

def dim(m):
    return len(m.shape)

def canDot(a, b):
    return dim(a) == 1 and dim(b) == 1 and a.shape == b.shape

def canMultiply(a, b):
    if dim(a) == 2 and dim(b) in [1, 2]:
        return a.shape[1] == b.shape[0]
    elif dim(a) == 1 and dim(b) in [1, 2]:
        return a.shape[0] == b.shape[0]
    else:
    	raise Exception('Multiplication not supported')

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
    assert(predictions.shape == targets.shape)
    return np.sqrt(np.mean(np.square(predictions-targets)))


def predictRMSE(f, x, y, kind=None):
    # Predictions of the train data
    y_predictions = f(x)
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
def axisDerivative(f, x, i, D):
    assert(x.shape == (D,))
    epsilon = 0.1
    epsilon_matrix = np.zeros(D)
    epsilon_matrix[i] += epsilon

    ret = (f(x + epsilon_matrix) - f(x - epsilon_matrix)) / (2 * epsilon)
    assert(np.isscalar(ret))
    return ret

# f is a mapping (D,) -> 1 and grad is a mapping (D,) -> (D,)
def testGradient(f, grad, D):
    test = np.random.rand(D)
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

def QRRegression(X_train, y_train):
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
