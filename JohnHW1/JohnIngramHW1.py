# John Ingram 
# CS 430 Homework 1

import matplotlib.pyplot as plt

# Implement the functions from the lecture slides

# Implement "the J function"
def J(theta0, theta1, m, x, y):
    return (1 / (2 * m)) * sum([(theta0 + theta1 * x[i] - y[i]) ** 2 for i in range(m)])

# implement y hat
def y_hat(theta0, theta1, x, i):
    return theta0 + theta1 * x[i]

# implement y hat - y
def y_hat_minus_y(theta0, theta1, x, y, i):
    return y_hat(theta0, theta1, x, i) - y[i]

# implement y hat - y times x
def y_hat_minus_y_times_x(theta0, theta1, x, y, i):
    return y_hat_minus_y(theta0, theta1, x, y, i) * x[i]

# Tie it all together and implement the gradient descent algorithm
def gradient_descent(theta0, theta1, alpha, epsilon, m, x, y):
    while True:
        # calculate the new theta values
        theta0_new = theta0 - alpha * (1 / m) * sum([y_hat_minus_y(theta0, theta1, x, y, i) for i in range(m)])
        theta1_new = theta1 - alpha * (1 / m) * sum([y_hat_minus_y_times_x(theta0, theta1, x, y, i) for i in range(m)])

        # calculate the change in cost
        delta_cost = abs(J(theta0_new, theta1_new, m, x, y) - J(theta0, theta1, m, x, y))
        
        # check if the error is acceptable
        if delta_cost > epsilon:
            # if the error is not acceptable, update the theta values and continue
            theta0 = theta0_new
            theta1 = theta1_new
            continue
        else:
            # if the error is acceptable, return the theta values
            return theta0, theta1

# open the file and read the data into lists
with open('data.txt') as f:
    # read the data into a list of strings
    data = f.read().splitlines()

    # split the data into a list of tuples and convert the values to floats
    data = [tuple(map(float, line.split(','))) for line in data]

    # unpack the tuples into separate x and y lists
    x, y = zip(*data)

# plot the data as a scatter plot
plt.plot(x, y, 'rx', alpha=0.5, label='Training Data')
plt.axis( xmin=4, xmax=24, ymin=-5, ymax=25)
plt.xticks(range(4, 25, 2))
plt.yticks(range(-5, 26, 5))
plt.xlabel('City Population in 10,000s')
plt.ylabel('Profit in $10,000s')

# run gradient descent to find the line of best fit
theta0, theta1 = gradient_descent(0, 0, 0.01, 1E-10, len(x), x, y)

# print the theta values
print('theta0 =', theta0)
print('theta1 =', theta1)

# Using the theta values, calculate the predicted profit for a city with a population of 35,000 and 70,000
print(f'For a city with a population of 35,000, the predicted profit is ${((theta0 + theta1 * 3.5)*10000):.2f}')
print(f'For a city with a population of 70,000, the predicted profit is ${((theta0 + theta1 * 7)*10000):.2f}')

# calculate the y values for the line of best fit
y_fit = [theta0 + theta1 * x[i] for i in range(len(x))]

# plot the line of best fit 
plt.plot(x, y_fit, 'b-', alpha=0.5, label='Gradient Descent')

# show the plot with a legend
plt.legend(loc='lower right')
plt.show()