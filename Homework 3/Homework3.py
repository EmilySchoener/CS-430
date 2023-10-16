#---Homework 3---
import numpy as np
import sigmoid as sigmoid
from logistricRegression import logisticRegression
from sklearn.metrics import confusion_matrix


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

# Create a list of classes
classes = [1, 2, 3]

for clss in classes:
    # Train to verify clss vs 2 other classes
    other_classes = [c for c in classes if c != clss]
    ATheta = logisticRegression(
        np.array(train)[:, :4], (np.array(train)[:, 4] == clss).astype(int))

    print(f"Class {clss} vs Classes {other_classes}, Thetas: ", ATheta)

    y = np.array(verify)[:, 4]
    x = np.array(verify)[:, :4]

    # Predict yhat
    yhat = x @ ATheta.T
    p = []
    for h in yhat:
        p.append(sigmoid.sigmoid(h))
    yPred = []
    for t in p:
        if t > 0.5:
            yPred.append(1)
        else:
            yPred.append(0)

    # Validation accuracy
    #tp = true positive, fp = false positive, tn = true negative, fn = false negative
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # Identify if y was correctly predicted
    for i in range(len(y)):
        if yPred[i] == 1 and y[i] == clss:
            tp += 1
        elif yPred[i] == 1 and y[i] not in [clss]:
            fp += 1
        elif yPred[i] == 0 and y[i] == clss:
            fn += 1
        elif yPred[i] == 0 and y[i] not in [clss]:
            tn += 1

    # Calculating accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    # Calculating Precision
    pre = tp / (tp + fp)

        # Calculate the confusion matrix
    cm = confusion_matrix(y, yPred)

    # Print the confusion matrix
    print(f"Confusion Matrix for class {clss} vs {other_classes}:")
    print(cm)
    print(f"Accuracy for class {clss} vs {other_classes} = {acc} and Precision = {pre}\n\n")
