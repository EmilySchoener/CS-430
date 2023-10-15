#---Homework 3---
import numpy as np
import sigmoid as sigmoid
from logistricRegression import logisticRegression


file = "iris.txt"
np.random.seed(20)


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

# Turn all the data into floats
for i in range(len(setosa)):
    setosa[i] = [float(x) for x in setosa[i]]

for i in range(len(versicolor)):
    versicolor[i] = [float(x) for x in versicolor[i]]

for i in range(len(virginica)):
    virginica[i] = [float(x) for x in virginica[i]]
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

#Train to verify Iris-setosa based on Iris-versicolor and Iris-virginica

theta = logisticRegression(np.array(train)[:,:4], np.array(train)[:,4])

print("Theta: ", theta)


#print(train[0])
#print(verify[0])
#print(setosa[0])
#print(versicolor[0])
#print(virginica[0])
