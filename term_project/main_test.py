import csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras

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
print("\n\n")


# Creating a neural net with 12 input nodes : 'X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', and 'rain'.
# The output is a singular node : 'area'
# There are two hidden layers - one with 128 nodes and the other with 64 nodes
# Create a neural network
model = keras.Sequential([
    keras.layers.Input(shape=(12,)),  # Input shape: 12 features
    keras.layers.Dense(128, activation='relu'), #Rectified Linear Unit for hidden layer activiaiton
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Output layer for regression
])

# Compile the neural net
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural net
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the neural net
y_pred_nn = model.predict(np.array(X_test))
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("Neural Network Results:")
print(f"Mean Squared Error: {mse_nn}")
print(f"R-squared Score: {r2_nn}")


"""
#using cross-validatrion to see results
# Number of folds for cross-validation
n_folds = 5

# Initialize variables to store results
mse_scores = []
r2_scores = []

# Perform K-fold cross-validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=600)
for train_indices, val_indices in kf.split(X):
    X_train_cv, X_val_cv = np.array(X)[train_indices], np.array(X)[val_indices]
    y_train_cv, y_val_cv = np.array(y)[train_indices], np.array(y)[val_indices]

    # Create a new neural network model for each fold
    model = keras.Sequential([
        keras.layers.Input(shape=(12,)),  # Input shape: 12 features
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on the training data for this fold
    model.fit(X_train_cv, y_train_cv, epochs=10, batch_size=32, validation_data=(X_val_cv, y_val_cv))

    # Evaluate the model on the validation data for this fold
    y_pred_cv = model.predict(X_val_cv)
    mse_cv = mean_squared_error(y_val_cv, y_pred_cv)
    r2_cv = r2_score(y_val_cv, y_pred_cv)

    # Store the cross-validation results
    mse_scores.append(mse_cv)
    r2_scores.append(r2_cv)

# Calculate and print the average and standard deviation of the cross-validation results
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print("Cross-Validation Results:")
print(f"Average Mean Squared Error: {avg_mse} (Std: {std_mse})")
print(f"Average R-squared Score: {avg_r2} (Std: {std_r2}")

"""

#32/16 nodes for hidden layer performed worse than 64/32.
#128/64 showed even better performance, but little to no real improvement in increasing nodes/layers past this