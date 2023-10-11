#---Homework 3---
import numpy as np

file = "iris.data"

data = []

# Read in data from file
with open(file) as f:
    for line in f:
        if line == '\n':
            break
        line = line.strip()
        l = line.split(',')
        if l[4] == "Iris-setosa":
            l[4] = 1
        elif l[4] == "Iris-versicolor":
            l[4] = 2
        elif l[4] == "Iris-virginica":
            l[4] = 3
        data.append(l)
np.random.shuffle(data)
print(data[0])
