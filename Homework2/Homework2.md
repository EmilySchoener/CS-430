**CS 430/530 - Assignment 2**

**Due date – Sunday, October 1**** st ****, 2023, on or before 11:59 pm.**

**Assigned - Sunday, September 17**** th ****, 2023**

**Please answer all the questions. No late submissions will be accepted. The solutions for all questions except the programming project should be neatly typed or written on a document and submitted through Canvas on or before the deadline. It can also be submitted in class in hard copy before the deadline.**

**Problem 01 (Programming project) [200 points]**

**This problem needs to be submitted separately through email. See submission section below.**

# Problem statement: Linear regression with multiple features

#

In this exercise, you will implement linear regression with multiple variables to

1. Predict the nitric oxides concentration (NOX)
2. Predict the prices of houses (MEDV)

For the "The Boston house-price" dataset. The link for the dataset and its description is shared below:

[http://lib.stat.cmu.edu/datasets/boston](http://lib.stat.cmu.edu/datasets/boston). Also the data is provided on canvas as "boston.txt"

**The targeted tasks are as follows:**

1. **Read the data description carefully. In this assignment you will create a training set with 90% of the data and a validation set with remaining 10% of the data. Note that the last 50 rows (~ 10%) of the dataset will be treated as the validation set for this homework.**

1. **You will train a regressor on the training set for the following:**
  1. **Predicting NOX based on DIS and RAD**
  2. **Predicting NOX based rest of the variables.**
  3. **Predicting MEDV based on AGE and TAX.**
  4. **Predicting MEDV based on the rest of the variables.**

**Feature Normalization**

You start by loading values from the given dataset. Carefully split the dataset between training and validation sets. If needed, please normalize the data by:

• Subtracting the mean value of each feature from the dataset.

• After subtracting the mean, additionally scale (divide) the feature values by their respective "standard deviations."

**Part-1: Gradient Descent**

For all the aforementioned 4 tasks estimate the parameters using gradient descent. Previously, you implemented gradient descent on a univariate regression problem. The difference now is that there are more features in the matrix X. The hypothesis function and the batch gradient descent update rules remain unchanged.

You can use your previous code with necessary modifications to implement the cost function and gradient descent for linear regression with multiple variables. You can keep the learning rate and convergence criterion same as the previous one.

Once you have found θ using this method you will predict the target variable (NOX and MEDV) based on the cases mentioned above by selecting the appropriate variables from the validation set. For example, for NOX first you will predict the 50 NOX values on the validation set using values of DIS and RAD from the validation set. For the 50 predicted values compute the sum of square error between the predicted and the given true values. Do the same thing for NOX with all the variables. The same task should be performed for MEDV. Please report the total squared error for each of the four cases.

**Implementation Note:** Please remember you will have to use the same normalization technique on the validation set that you used on the training set.

**Part-2: Normal Equations**

You have also learned that the closed-form solution to linear regression is

θ = (X T X) − 1 X T y

Your goal is to calculate θ only for cases 2a and 2c using the formula above. Remember that while you don't need to scale your features, you still need to add a column of 1's to the X matrix to have an intercept term (θ0).

Now, once you have found θ using this method, use it to calculate the total squared error for both the cases using the method described for Part-1 ( Gradient Descent).

**Programming languages and formatting**

You should use preferably python or MATLAB and should implement the algorithms yourself. You can use available library functions to calculate inverse and transpose of the matrices and for matrix multiplications as well.

Please do NOT use any library function which provides an implementation of linear regression or gradient descent. You will get ZERO otherwise.

**Submission Guidelines**

Your submission must have the following:

- A **README file** that describes how the code can be compiled and run. Also list any external dependencies that need to be satisfied for compiling and running the code. If your code fails to compile YOU GET A ZERO.
- An **output file** having the computed theta values and calculated squared error using both the techniques as required in the problem description should be submitted.
- Note that each group will submit only one solution for the programming project .

Please e-mail your submission to [hs0111@uah.edu](mailto:hs0111@uah.edu) by 11:59 PM on Sunday, October 1st, 2023. DO NOT SUBMIT THROUGH DROPBOX or CANVAS. They will not be accepted and will result in late penalty. Put all your materials for the PROGRAMMING project ONLY in a folder with your name and then create a zip file out of it. Email the zip file to the TA.

**Point Breakdown**

For loading the data: 10 points

Splitting the data into test set and validation set: 10 points

For correct implementation of gradient descent for 2a: 10 points

For correct implementation of gradient descent for 2b: 25 points

For correct implementation of gradient descent for 2c: 10 points

For correct implementation of gradient descent for 2d: 25 points

Computation of squared error in case of 2a: 10 points

Computation of squared error in case of 2b: 15 points

Computation of squared error in case of 2c: 10 points

Computation of squared error in case of 2d: 15 points

Computing theta values solving normal equations in case of 2a: 15 points

Computing theta values solving normal equations in case of 2c: 15 points

Computation of squared error in case of 2a: 10 points

Computation of squared error in case of 2c: 10 points

Making the output file: 10 points

For predictions: 5 points

For correct implementation of normal equations: 35 points

For predictions: 5 points

Output file: 5 points

4