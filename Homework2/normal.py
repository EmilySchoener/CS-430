    X = np.zeros((456, 2), dtype=float)
    X[:, 0] = tdata["DIS"]
    X[:, 1] = tdata["RAD"]
    Y = tdata["NOX"]
    
    X_one_a = np.column_stack((np.ones(X.shape[0]), X))
    n_theta_a = np.linalg.inv(X_one_a.T @ X_one_a) @ X_one_a.T @ Y #Doing the normal equation (X^TX)^-1 * X^T * y

#print(z_theta_a[0], z_theta_a[1], z_theta_a[2]) Uncomment to check values
#creating another Yhat to not mess with your other work
    Yhat_a = np.zeros((50,), dtype=float)
    for i in range(50):
        Yhat_a[i] = n_theta_a[0] * vdata['DIS'][i] + n_theta_a[1] * vdata['RAD'][i] + n_theta_a[2] #applying calculated thetas from normal

    print('Sum of squared errors for normal 2a =', sum((Yhat_a - vdata['NOX']) ** 2))

    
    X_c = np.zeros((456,2), dtype=float)
    X_c[:, 0] = tdata["AGE"]
    X_c[:, 1] = tdata["TAX"]
    Y_c = tdata["MEDV"]

    X_one_c = np.column_stack((np.ones(X_c.shape[0]), X_c))
    n_theta_c = np.linalg.inv(X_one_c.T @ X_one_c) @ X_one_c.T @ Y_c

#print(n_theta_c[0], n_theta_c[1], n_theta_c[2]) Uncomment to check values

    Yhat_c = np.zeros((50,), dtype=float)
    for i in range(50):
        Yhat_c[i] = n_theta_c[0] * vdata["AGE"][i] + n_theta_c[1] * vdata["TAX"][i] + n_theta_c[2]

    print('Sum of squared errors for normal 2c =', sum((Yhat_c - vdata["MEDV"]) ** 2))