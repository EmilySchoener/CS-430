#--CS 430 Homework 2--
import itertools
import numpy as np
import GradientDescent as gd

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

data = np.float64(data) #Change data type

#add column of 1s for X0
data = np.concatenate((np.ones((506,1), dtype=float), data), axis=1)

#Split into training data and verification data
train = data[:456]
verify = data[456:]

#--- Create Dictionaries ---
tdata = dict()
tdata['CRIM'] = train[:,1]
tdata['ZN'] = train[:,2]
tdata['INDUS'] = train[:,3]
tdata['CHAS'] = train[:,4]
tdata['NOX'] = train[:,5]
tdata['RM'] = train[:,6]
tdata['AGE'] = train[:,7]
tdata['DIS'] = train[:,8]
tdata['RAD'] = train[:,9]
tdata['TAX'] = train[:,10]
tdata['PTRATIO'] = train[:,11]
tdata['B'] = train[:,12]
tdata['LSTAT'] = train[:,13]
tdata['MEDV'] = train[:,14]

vdata = dict()
vdata['CRIM'] = verify[:,1]
vdata['ZN'] = verify[:,2]
vdata['INDUS'] = verify[:,3]
vdata['CHAS'] = verify[:,4]
vdata['NOX'] = verify[:,5]
vdata['RM'] = verify[:,6]
vdata['AGE'] = verify[:,7]
vdata['DIS'] = verify[:,8]
vdata['RAD'] = verify[:,9]
vdata['TAX'] = verify[:,10]
vdata['PTRATIO'] = verify[:,11]
vdata['B'] = verify[:,12]
vdata['LSTAT'] = verify[:,13]
vdata['MEDV'] = verify[:,14]
#--- End of Dictionary Creation ---

# Create matrices to train NOX based on DIS and RAD
X = np.zeros((456,2), dtype=float)
X[:,0] = tdata['DIS']
X[:,1] = tdata['RAD']
Y = tdata['NOX']

# run gradient descent to find the line of best fit
alpha = 0.01
epsilon = 1e-4
theta = gd.gradient_descent(X, Y, alpha, epsilon)
print('theta =', theta)

# Calculate the predicted NOX for the verification data
# and report the sum of the squared errors
# TODO: FIX THIS, I AM ALMOST CERTAIN IT IS WRONG
Yhat = np.zeros((50,), dtype=float)
for i in range(50):
    Yhat[i] = theta[0] * vdata['DIS'][i] + theta[1] * vdata['RAD'][i] + theta[2]

print('Sum of squared errors =', sum((Yhat - vdata['NOX']) ** 2))