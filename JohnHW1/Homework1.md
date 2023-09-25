**CS 430/530 - Assignment 1**

**Due date – Wednesday, September 13**** th ****, 2023, on or before 11:59 pm.**

**Assigned - Sunday, September 3**** rd ****, 2023**

**Please answer all the questions. No late submissions will be accepted. The solutions for all questions except the programming project should be neatly typed or written on a document and submitted through Canvas on or before the deadline. It can also be submitted in class in hard copy before the deadline.**

**Please write your name legibly and staple the sheets before submitting.**

**Problem**** 01 [5 +5+5+5= 20 points]**

1. In linear regression the cost function is expressed as:

2

Please explain the reason why a squared error is used in the cost fuction instead of an absolute error.

1. Use the least square method to solve for the parameters & for linear regression and compute the values of the parameters for given data set. Compare the values you got using least square method with the values that you will get using gradient descent ( see the programming assignment below). How close are they?

**Problem 02 (Programming project) [80 points]**

**This problem needs to be submitted separately through email. See submission section below.**

# Problem statement: Linearregressionwithonevariable

#

In this exercise, you will implement linear regression with one variable to predict profit for a retail store.Suppose you are the CEO of a retail store chain and are considering different cities for opening a new outlet.The chain already has stores in various cities, and you have data for profits for each of those stores and the population of the corresponding city.

Thefiledata.txtcontainsthedatasetforourlinearregressionproblem.The first column is the population of a city, and the second column is the profit of the retail store in that city.A negative value for profit indicates a loss.

## **Plotting**** the ****Data**

Before starting on any task, it is often useful to understand the data by visualizing it.For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population).Now, when you plot the data, the result should look like Figure [1,](#_bookmark1) with the same red "x" markers and axis labels.

![](RackMultipart20230913-1-k6jmuh_html_e0189be2e392e1ba.png)

## **Gradient**** Descent**

As discussed in class, model the problem as a linear regression problem with a single variable and estimate the parameters using gradient descent. Hence, youwillfitthelinearregressionparameters & toourdataset using gradient descent. Use the cost function and the update equations (of the parameters) as discussed in class and are available in slides. you will use batch gradient descent to solve the problems.

Initialize the initial parameters to 0 and the learning rate alphato 0.01. Agoodwaytoverifythatgradientdescentisworkingcorrectlyistolook atthevalueofJ()andcheckthatitisdecreasingwitheachstep.Your program should stop after reaching an acceptable accuracy, say or less.

Afteryouare finished, use your final parameters to plot the linear fit.The result should look something like Figure [2:](#_bookmark2)

![](RackMultipart20230913-1-k6jmuh_html_ff7ee18cda2310a3.png)

Finally, use the finalvaluesfor & tomakepredictionsonprofitsin areas of 35,000 and 70,000 people.

**Programming languages and formatting**

You should use preferably python or MATLAB and should implement the algorithms yourself. Please do NOT use any library function which provides an implementation of linear regression or gradient descent. You will get ZERO otherwise.

**Submission Guidelines**

Your submission must have the following:

A README file that describes how the code can be compiled and run. Also list any external dependencies that need to be satisfied for compiling and running the code. If your code fails to compile YOU GET A ZERO.

Please e-mail your submission to [hs0111@uah.edu](mailto:hs0111@uah.edu) by 11:59 PM on Wednesday September 13th, 2023. DO NOT SUBMIT THROUGH DROPBOX or CANVAS. They will not be accepted and will result in late penalty. Put all your materials for the PROGRAMMING project ONLY in a folder with your name and then create a zip file out of it. Email the zip file to the TA.

**Point Breakdown**

For loading the data: 10 points

For plotting the data: 10 points

For correct implementation of gradient descent: 40 points

For plotting the regression line: 10 points

For predictions: 5 +5 = 10 points

3