import numpy as np
import sigmoid as sigmoid

def logistic_regression_cost(theta, X, y):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid.sigmoid(z)
    cost = -1/m * np.sum(y * np.log(h + .00000000001) + (1 - y) * np.log(1 - h + .00000000001))
    return cost