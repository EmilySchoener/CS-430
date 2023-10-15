import numpy as np
import sigmoid as sigmoid
from lrc import logistic_regression_cost
from scipy.optimize import minimize

# Function to run logistic regression on the supplied data
# Uses scipy's minimize function to find the optimal theta values
# The logistic regression cost function is minimized using the BFGS algorithm
def logisticRegression(X, y):

    x0 = np.zeros(X.shape[1])

    # Run minimize to obtain the optimal theta
    # This function will return theta and the cost
    result = minimize(logistic_regression_cost, x0, args=(X, y), method='BFGS', options={"maxiter":500})
    
    return result.x
