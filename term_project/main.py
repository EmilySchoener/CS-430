import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

# Converts the months and days into ints to use for the training models
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

day_mapping = {
    'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7
}

data = []
with open('forestfires.csv', mode='r') as file:
    csvFile = csv.DictReader(file)
    for row in csvFile:
        data.append(row)

# targets the one feature (area) against all other traits in the data set
features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
target = 'area'

# Applies a logrithm transform as was suggested by the data set introduciton
y = [np.log(1 + float(row[target])) for row in data]

# Extract and convert the 'month' and 'day' to numerical values
X = []
for row in data:
    x = [
        float(row['X']),
        float(row['Y']),
        month_mapping[row['month']],
        day_mapping[row['day']],
        float(row['FFMC']),
        float(row['DMC']),
        float(row['DC']),
        float(row['ISI']),
        float(row['temp']),
        float(row['RH']),
        float(row['wind']),
        float(row['rain'])
    ]
    X.append(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=600)

# Create and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Linear predictions
y_pred_linear = linear_model.predict(X_test)

# Create and train the SVR model
svr_model = SVR(kernel='rbf')  # The 'rbf' kernal was chosen since there is no prior knowledge on how the data's underlying distribution
svr_model.fit(X_train, y_train)

# Predicts using SVR
y_pred_svr = svr_model.predict(X_test)

# Linear performance
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# SVR performance
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("Linear Regression Results:")
print(f"Mean Squared Error: {mse_linear}")
print(f"R-squared Score: {r2_linear}")

print("\nSupport Vector Regression (SVR) Results:")
print(f"Mean Squared Error: {mse_svr}")
print(f"R-squared Score: {r2_svr}")