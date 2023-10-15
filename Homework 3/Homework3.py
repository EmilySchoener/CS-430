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
clss = 1
ATheta = logisticRegression(np.array(train)[:,:4], np.array(train)[:,4])

print("Class 1 vs 2 and 3, Theta: ", ATheta)

y = np.array(verify)[:,4]
x = np.array(verify)[:,:4]

#Predict yhat
yhat = x @ ATheta.T
p = []
for h in yhat:
    p.append(sigmoid.sigmoid(h))
yPred =[]
for t in p:
    if t > 0.5:
        yPred.append(1)
    else:
        yPred.append(0)
               
#print(yPred)
#print(y)

#Validation accuracy for A
tp = 0
fp = 0
tn = 0
fn = 0
#Identify if y was correctly predicted
for i in range(len(y)):
    if yPred[i] == 1 and y[i] == clss:
        tp += 1
    elif yPred[i] == 1 and y[i] != clss:
        fp += 1
    elif yPred[i] == 0 and y[i] == clss:
        fn += 1
    elif yPred[i] == 0 and y[i] != clss:
        tn += 1
#Accuracy
acc = (tp + tn)/(tp+tn+fp+fn)
#Precision
pre = tp / (tp+fp)
print("Accuracy = " + str(acc) + " and Precision = " + str(pre) +"\n\n")



#print(train[0])
#print(verify[0])
#print(setosa[0])
#print(versicolor[0])
#print(virginica[0])
