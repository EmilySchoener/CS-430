# --CS 430 Homework 2--
import itertools
import numpy as np
import GradientDescent as gd

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

# Function to normalize a column of data
def normalize(data):
    std_dev = np.std(data)
    if std_dev == 0:
        std_dev = 1

    return (data - np.mean(data)) / std_dev


# Function to print a list of thetas
def print_thetas(thetas):
    for i in range(len(thetas)):
        print(f"\tθ{i} = {thetas[i]}")


# ----------------- Problem 1 -----------------
gd.do_gradient_descent()

# This should be right, but i think the sum of squared errors is wrong for 2c...
# I'll look more into it.

# Create matrix with first column of 1's
X_one_a = np.column_stack((np.ones(X.shape[0]), X))
n_theta_a = (
    np.linalg.inv(X_one_a.T @ X_one_a) @ X_one_a.T @ Y
)  # Doing the normal equation (X^TX)^-1 * X^T * y

# print(z_theta_a[0], z_theta_a[1], z_theta_a[2]) Uncomment to check values
# creating another Yhat to not mess with your other work
Yhat_a = np.zeros((50,), dtype=float)
for i in range(50):
    Yhat_a[i] = (
        n_theta_a[0] * vdata["DIS"][i] + n_theta_a[1] * vdata["RAD"][i] + n_theta_a[2]
    )  # applying calculated thetas from normal

print("Sum of squared errors for normal 2a =", sum((Yhat_a - vdata["NOX"]) ** 2))

X_c = np.zeros((456, 2), dtype=float)
X_c[:, 0] = tdata["AGE"]
X_c[:, 1] = tdata["TAX"]
Y_c = tdata["MEDV"]

X_one_c = np.column_stack((np.ones(X_c.shape[0]), X_c))
n_theta_c = np.linalg.inv(X_one_c.T @ X_one_c) @ X_one_c.T @ Y_c

# print(n_theta_c[0], n_theta_c[1], n_theta_c[2]) Uncomment to check values

Yhat_c = np.zeros((50,), dtype=float)
for i in range(50):
    Yhat_c[i] = (
        n_theta_c[0] * vdata["AGE"][i] + n_theta_c[1] * vdata["TAX"][i] + n_theta_c[2]
    )

print("Sum of squared errors for normal 2c =", sum((Yhat_c - vdata["MEDV"]) ** 2))
