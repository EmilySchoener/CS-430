# Implement multivariate linear regression using gradient descent
# The data will be in numpy matrix format, so we will use 
# numpy matrix operations to implement the functions.

import numpy as np

def gradient_descent(X, Y, alpha, epsilon):
    m = len(X)

    # Add a column of ones to the X matrix for the intercept term
    ones=np.ones(len(X))
    features = np.c_[ones, X]

    # Initialize theta to a vector of zeros
    theta = np.zeros(features.shape[1])

    while True:
        theta_new = theta.copy()

        # Make a prediction with the current theta values
        y_hat = np.dot(features, theta.T)

        # Subtract the actual value from the prediction to get the error ( y_hat - Y )
        error = y_hat - Y

        # Calculate the new theta values
        theta_new = (alpha * 1/m * np.dot(features.T, error))

        theta_new = theta - theta_new

        # Subtract the new theta values from the old theta values to get the change in theta ( theta_new - theta )
        delta = theta_new - theta

        # Take the absolute value of the change in theta
        delta = np.abs(delta)
        
        # Check if the error is acceptable
        if np.all(delta > epsilon):
            # If the error is not acceptable, update the theta values and continue
            theta = theta_new
            continue
        else:
            # If the error is acceptable, return the theta values
            return theta_new
        
