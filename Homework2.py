#--CS 430 Homework 2--
import itertools
import numpy as np

file = "boston.txt"

data = []

#Read in data from file
with open(file) as f:
    for line in itertools.islice(f, 22, None): #ignore the first 22 lines in file
        temp = []
        l = line.split()
        for x in l:
            temp.append(x)
        line = next(f)
        l = line.split()
        for x in l:
            temp.append(x)
        data.append(temp)

data = np.matrix(np.array(data)) #Make data a numpy matrix
data = np.float64(data) #Change data type

#Test
#print(data[0])
#print(data.shape)

#Split into training data and varification data
train = data[:456]
varify = data[456:]

#Test to make sure it split correctly
#print(train[455])
#print(varify[0])
