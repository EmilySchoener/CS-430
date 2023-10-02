# Implement multivariate linear regression using gradient descent
# also implement the normal equation method

# Written by: Emily Schoener, Josh Velazquez, and John Ingram

import numpy as np
import itertools
import sys

def do_gradient_descent():

    file = "boston.txt"

    data = []

    # Read in data from file
    with open(file) as f:
        for line in itertools.islice(f, 22, None):  # ignore the first 22 lines in file
            temp = []
            l = line.split()
            for x in l:
                temp.append(x)
            line = next(f)
            l = line.split()
            for x in l:
                temp.append(x)
            data.append(temp)

    data = np.float64(data)  # Change data type

    # add column of 1s for X0
    data = np.concatenate((np.ones((506, 1), dtype=float), data), axis=1)

    # Split into training data and verification data
    train = data[:456]
    verify = data[456:]

    # --- Create Dictionaries ---
    tdata = dict()
    tdata["CRIM"] = train[:, 1]
    tdata["ZN"] = train[:, 2]
    tdata["INDUS"] = train[:, 3]
    tdata["CHAS"] = train[:, 4]
    tdata["NOX"] = train[:, 5]
    tdata["RM"] = train[:, 6]
    tdata["AGE"] = train[:, 7]
    tdata["DIS"] = train[:, 8]
    tdata["RAD"] = train[:, 9]
    tdata["TAX"] = train[:, 10]
    tdata["PTRATIO"] = train[:, 11]
    tdata["B"] = train[:, 12]
    tdata["LSTAT"] = train[:, 13]
    tdata["MEDV"] = train[:, 14]

    vdata = dict()
    vdata["CRIM"] = verify[:, 1]
    vdata["ZN"] = verify[:, 2]
    vdata["INDUS"] = verify[:, 3]
    vdata["CHAS"] = verify[:, 4]
    vdata["NOX"] = verify[:, 5]
    vdata["RM"] = verify[:, 6]
    vdata["AGE"] = verify[:, 7]
    vdata["DIS"] = verify[:, 8]
    vdata["RAD"] = verify[:, 9]
    vdata["TAX"] = verify[:, 10]
    vdata["PTRATIO"] = verify[:, 11]
    vdata["B"] = verify[:, 12]
    vdata["LSTAT"] = verify[:, 13]
    vdata["MEDV"] = verify[:, 14]
    # --- End of Dictionary Creation ---


    def gradient_descent(X, Y, alpha, epsilon):
        m = len(X)

        # Add a column of ones to the X matrix for the intercept term
        ones = np.ones(len(X))
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
            theta_new = alpha * 1 / (2 * m) * np.dot(features.T, error)

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

    # Function to normalize a column of data
    def normalize(data):
        std_dev = np.std(data)
        if std_dev == 0:
            std_dev = 1

        return (data - np.mean(data)) / std_dev

    # Function to print a list of thetas
    def print_thetas(thetas):
        for i in range(len(thetas)):
            print(f"\tTheta:{i} = {thetas[i]}")

    # ----------------- Problem 1 -----------------
    alpha = 0.01
    epsilon = 1e-4
    # ____________1____________
    print("Problem 1.a: Predict NOX from DIS and RAD")
    # Create matrices to train NOX based on DIS and RAD
    X = np.zeros((456, 2), dtype=float)
    X[:, 0] = tdata["DIS"]
    X[:, 1] = tdata["RAD"]

    Y = tdata["NOX"]

    # run gradient descent to find the line of best fit
    theta = gradient_descent(X, Y, alpha, epsilon)
    print_thetas(theta)

    # Calculate the predicted NOX for the verification data
    # and report the sum of the squared errors
    X = np.zeros((50, 2), dtype=float)
    X[:, 0] = vdata["DIS"]
    X[:, 1] = vdata["RAD"]
    ones = np.ones(len(X))
    X = np.c_[ones, X]
    h = np.dot(X, theta.T)
    result = 0
    for i in range(50):
        result += (h[i] - vdata["NOX"][i]) ** 2

    print("\tSum of squared errors =", result)
    print("\n\n")


    # ____________2____________
    print("Problem 1.b: Predict NOX From all other variables")
    # Create matrices to train NOX based on all other variables
    X = np.zeros((456, 13), dtype=float)
    X[:, 0] = tdata["CRIM"]
    X[:, 1] = tdata["ZN"]
    X[:, 2] = tdata["INDUS"]
    X[:, 3] = tdata["CHAS"]
    X[:, 4] = tdata["RM"]
    X[:, 5] = tdata["AGE"]
    X[:, 6] = tdata["DIS"]
    X[:, 7] = tdata["RAD"]
    X[:, 8] = tdata["TAX"]
    X[:, 9] = tdata["PTRATIO"]
    X[:, 10] = tdata["B"]
    X[:, 11] = tdata["LSTAT"]
    X[:, 12] = tdata["MEDV"]

    # Normalize the data
    for i in range(13):
        X[:, i] = normalize(X[:, i])

    Y = normalize(tdata["NOX"])

    # run gradient descent to find the line of best fit
    theta = gradient_descent(X, Y, alpha, epsilon)
    print_thetas(theta)

    # Create matrix out of verification data
    X = np.zeros((50, 13), dtype=float)
    X[:, 0] = vdata["CRIM"]
    X[:, 1] = vdata["ZN"]
    X[:, 2] = vdata["INDUS"]
    X[:, 3] = vdata["CHAS"]
    X[:, 4] = vdata["RM"]
    X[:, 5] = vdata["AGE"]
    X[:, 6] = vdata["DIS"]
    X[:, 7] = vdata["RAD"]
    X[:, 8] = vdata["TAX"]
    X[:, 9] = vdata["PTRATIO"]
    X[:, 10] = vdata["B"]
    X[:, 11] = vdata["LSTAT"]
    X[:, 12] = vdata["MEDV"]

    for i in range(13):
        X[:, i] = normalize(X[:, i])

    ones = np.ones(len(X))
    X = np.c_[ones, X]
    h = np.dot(X, theta.T)
    result = 0
    for i in range(50):
        result += (h[i] - normalize(vdata["NOX"])[i]) ** 2

    print("\tSum of squared errors =", result)
    print("\n\n")


    # ____________3____________
    print("Problem 1.c: Predict MEDV from AGE and TAX")
    # Create matrices to train MEDV based on AGE and TAX
    X = np.zeros((456, 2), dtype=float)
    X[:, 0] = normalize(tdata["AGE"])
    X[:, 1] = normalize(tdata["TAX"])

    Y = normalize(tdata["MEDV"])

    # run gradient descent to find the line of best fit
    theta = gradient_descent(X, Y, alpha, epsilon)
    print_thetas(theta)

    # Create matrix out of verification data
    X = np.zeros((50, 2), dtype=float)
    X[:, 0] = normalize(vdata["AGE"])
    X[:, 1] = normalize(vdata["TAX"])
    ones = np.ones(len(X))
    X = np.c_[ones, X]
    h = np.dot(X, theta.T)
    result = 0
    for i in range(50):
        result += (h[i] - normalize(vdata["MEDV"])[i]) ** 2

    print("\tSum of squared errors =", result)
    print("\n\n")
    
    

    # ____________4____________
    print("Problem 1.d: Predict MEDV from all other variables")
    # Create matrices to train MEDV based on all other variables

    X = np.zeros((456, 13), dtype=float)
    X[:, 0] = tdata["CRIM"]
    X[:, 1] = tdata["ZN"]
    X[:, 2] = tdata["INDUS"]
    X[:, 3] = tdata["CHAS"]
    X[:, 4] = tdata["RM"]
    X[:, 5] = tdata["AGE"]
    X[:, 6] = tdata["DIS"]
    X[:, 7] = tdata["RAD"]
    X[:, 8] = tdata["TAX"]
    X[:, 9] = tdata["PTRATIO"]
    X[:, 10] = tdata["B"]
    X[:, 11] = tdata["LSTAT"]
    X[:, 12] = tdata["NOX"]

    # Normalize the data
    for i in range(13):
        X[:, i] = normalize(X[:, i])

    Y = normalize(tdata["MEDV"])

    # run gradient descent to find the line of best fit
    theta = gradient_descent(X, Y, alpha, epsilon)
    print_thetas(theta)

    # Create matrix out of verification data
    X = np.zeros((50, 13), dtype=float)
    X[:, 0] = vdata["CRIM"]
    X[:, 1] = vdata["ZN"]
    X[:, 2] = vdata["INDUS"]
    X[:, 3] = vdata["CHAS"]
    X[:, 4] = vdata["RM"]
    X[:, 5] = vdata["AGE"]
    X[:, 6] = vdata["DIS"]
    X[:, 7] = vdata["RAD"]
    X[:, 8] = vdata["TAX"]
    X[:, 9] = vdata["PTRATIO"]
    X[:, 10] = vdata["B"]
    X[:, 11] = vdata["LSTAT"]
    X[:, 12] = vdata["NOX"]

    for i in range(13):
        X[:, i] = normalize(X[:, i])

    ones = np.ones(len(X))
    X = np.c_[ones, X]
    h = np.dot(X, theta.T)
    result = 0
    for i in range(50):
        result += (h[i] - normalize(vdata["MEDV"])[i]) ** 2

    print("\tSum of squared errors =", result)
    print("\n\n")


    # ----------------- Problem 2 -----------------
    print("Problem 2: Use Normal Equations for 2a and 2c:")
    X = np.zeros((456, 2), dtype=float)
    X[:, 0] = tdata["DIS"]
    X[:, 1] = tdata["RAD"]
    Y = tdata["NOX"]
    
    X_one_a = np.column_stack((np.ones(X.shape[0]), X))
    n_theta_a = np.linalg.inv(X_one_a.T @ X_one_a) @ X_one_a.T @ Y #Doing the normal equation (X^TX)^-1 * X^T * y

    print_thetas(n_theta_a)
    #creating another Yhat to not mess with your other work
    Yhat_a = np.zeros((50,), dtype=float)
    for i in range(50):
        Yhat_a[i] = n_theta_a[0] + (vdata['DIS'][i] * n_theta_a[1]) + (vdata['RAD'][i] * n_theta_a[2]) #applying calculated thetas from normal

    print('\tSum of squared errors for normal 2a =', sum((Yhat_a - vdata['NOX']) ** 2))


    X_c = np.zeros((456,2), dtype=float)
    X_c[:, 0] = tdata["AGE"]
    X_c[:, 1] = tdata["TAX"]
    Y_c = tdata["MEDV"]

    X_one_c = np.column_stack((np.ones(X_c.shape[0]), X_c))
    n_theta_c = np.linalg.inv(X_one_c.T @ X_one_c) @ X_one_c.T @ Y_c
    print("\n")
    print_thetas(n_theta_c)

    Yhat_c = np.zeros((50,), dtype=float)
    for i in range(50):
        Yhat_c[i] = n_theta_c[0] + (vdata["AGE"][i] * n_theta_c[1]) + (vdata["TAX"][i] * n_theta_c[2])
    

    print('\tSum of squared errors for normal 2c =', sum((Yhat_c - vdata["MEDV"]) ** 2))


if __name__ == "__main__":
    do_gradient_descent()
    sys.stdout = open("output.txt", "w")
    do_gradient_descent()
    sys.stdout.close()