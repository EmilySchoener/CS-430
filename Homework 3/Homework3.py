#---Homework 3---
import numpy as np
import sigmoid as sigmoid
import lrc as lrc
from scipy.optimize import minimize

file = "iris.txt"

data = []
train = []
verify = []
setosa = []
versicolor = []
virginica = []


# Read in data from file
with open(file) as f:
    for line in f:
        if line == '\n':
            break
        line = line.strip()
        l = line.split(',')
        if l[4] == "Iris-setosa":
            l[4] = 1
            setosa.append(l)
        elif l[4] == "Iris-versicolor":
            l[4] = 2
            versicolor.append(l)
        elif l[4] == "Iris-virginica":
            l[4] = 3
            virginica.append(l)
#data shuffled to separate into train/verify
np.random.shuffle(setosa)
np.random.shuffle(versicolor)
np.random.shuffle(virginica)

#train verify split 80% 20%
train.extend(setosa[:40])
train.extend(versicolor[:40])
train.extend(virginica[:40])

verify.extend(setosa[-10:])
verify.extend(versicolor[-10:])
verify.extend(virginica[-10:])

# Prepare the training and verification data
X_train = np.array([row[:4] for row in train], dtype=float)
y_train = np.array([row[4] for row in train], dtype=int)

X_verify = np.array([row[:4] for row in verify], dtype=float)
y_verify = np.array([row[4] for row in verify], dtype=int)

initial_theta = np.zeros(X_train.shape[1])
result = minimize(lrc.logistic_regression_cost, initial_theta, args=(X_train, y_train), method='BFGS')

# The optimized parameters
optimized_theta = result.x

# Classification on the verification dataset
z = X_verify.dot(optimized_theta)
predictions = sigmoid.sigmoid(z)
predicted_classes = (predictions >= 0.5).astype(int)

# Evaluate the model's performance on the verification dataset
accuracy = np.mean(predicted_classes == y_verify)
print(f"Accuracy on verification dataset: {accuracy * 100}%")

#print(train[0])
#print(verify[0])
#print(setosa[0])
#print(versicolor[0])
#print(virginica[0])
