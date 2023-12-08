import csv
import numpy as np
import math
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

dontGraph = False
if len(sys.argv) > 1:
    if sys.argv[1] == "-ng":
        dontGraph = True

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
svr_model = SVR(kernel='rbf', max_iter=265)  # The 'rbf' kernal was chosen since there is no prior knowledge on the data's underlying distribution
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
print("\n")

# Get the coefficients (parameters)
coefficients = linear_model.coef_
bias_lr = linear_model.intercept_  

print("Linear Regression Coefficients:")
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef}")
print(f"Bias (Intercept): {bias_lr}")

# Calculate the SVR bias
bias_svr = np.mean(y_pred_svr - y_test)

print("\nSupport Vector Regression (SVR) Results:")
print(f"Mean Squared Error: {mse_svr}")
print(f"R-squared Score: {r2_svr}")
print(f"Bias: {bias_svr}")
print("\n")



#Using Random Forests
rf_model = RandomForestRegressor(n_estimators=100, random_state=600)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# Creating a neural net with 12 input nodes : 'X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', and 'rain'.
# The output is a singular node : 'area'
# There are two hidden layers - one with 128 nodes and the other with 64 nodes
model = keras.Sequential([
    keras.layers.Input(shape=(12,)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(.01)),
    keras.layers.Dense(1)
])

# Compile the neural net
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up of early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Train the neural net with early stopping
history = model.fit(np.array(X_train), np.array(y_train), epochs=1000, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the neural net
y_pred_nn = model.predict(np.array(X_test))
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

# Calculate the bias of the neural network
bias_nn = np.mean(y_pred_nn - y_test)

print("Neural Network Results:")
print(f"Mean Squared Error: {mse_nn}")
print(f"R-squared Score: {r2_nn}")
print(f"Bias: {bias_nn}")
print("\n")

# Get the weights of the neural network
weights = model.get_weights()

# Print the weights
print("Neural Network Weights:")
for i, layer_weights in enumerate(weights):
    print(f"Layer {i + 1}:")
    print(layer_weights)


# Inverse the log transformation for predicted values
y_pred_linear_original = np.exp(y_pred_linear) - 1
y_pred_svr_original = np.exp(y_pred_svr) - 1
y_pred_rf_original = np.exp(y_pred_rf) - 1
y_pred_nn_original = np.exp(y_pred_nn) - 1

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

#Calculate the bias of the random forest model
bias_rf = np.mean(y_pred_rf - y_test)

print("RF Results:")
print(f"Mean Squared Error: {mse_rf}")
print(f"R-squared Score: {r2_rf}")
print(f"Bias: {bias_rf}")


print("Summary of Results:")

# Print all of the r-squared scores for the models
print(f"Linear Regression R-squared Score: {abs(r2_linear)}")
print(f"SVR R-squared Score: {abs(r2_svr)}")
print(f"Random Forest R-squared Score: {abs(r2_rf)}")
print(f"Neural Network R-squared Score: {abs(r2_nn)}")

# Print all of the mean squared errors for the models
print(f"Linear Regression Mean Squared Error: {mse_linear}")
print(f"SVR Mean Squared Error: {mse_svr}")
print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Neural Network Mean Squared Error: {mse_nn}")

# Print all of the biases for the models
print(f"Linear Regression Bias: {bias_lr}")
print(f"SVR Bias: {bias_svr}")
print(f"Random Forest Bias: {bias_rf}")
print(f"Neural Network Bias: {bias_nn}")

if dontGraph:
    exit()


# Plot the predicted vs. actual values for the linear regression model
#plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, color='blue', label='Linear Regression')
plt.xlabel('Actual Log(Area)')
plt.ylabel('Predicted Log(Area)')
plt.title('Linear Regression: Actual vs. Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Plot the predicted vs. actual values for the SVR model
#plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_svr, color='red', label='SVR')
plt.xlabel('Actual Log(Area)')
plt.ylabel('Predicted Log(Area)')
plt.title('SVR: Actual vs. Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Random Forest predicted vs. actual values
plt.scatter(y_test, y_pred_rf, color='green', label='Random Forest')
plt.title('Random Forest: True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()

# Plot the predicted vs. actual values for the neural network
#plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_nn, color='orange', label='Neural Network')
plt.xlabel('Actual Log(Area)')
plt.ylabel('Predicted Log(Area)')
plt.title('Neural Network: Actual vs. Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Plot the estimated area for each feature based on the linear model
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features):
    # Create a subplot for each feature
    plt.subplot(4, 3, i+1)
    
    # Extract the current feature for plotting
    feature_values = [row[features.index(feature)] for row in X_test]
    
    # Plot the actual values against the predicted values
    plt.scatter(feature_values, y_test, color='blue', label='Actual')
    plt.scatter(feature_values, y_pred_linear, color='red', label='Predicted')
    
    plt.xlabel(feature)
    plt.ylabel('Log(Area)')
    plt.title(f'Linear Regression: Actual vs. Predicted for {feature}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the estimated area for each feature based on the SVR model
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features):
    # Create a subplot for each feature
    plt.subplot(4, 3, i+1)
    
    # Extract the current feature for plotting
    feature_values = [row[features.index(feature)] for row in X_test]
    
    # Plot the actual values against the predicted values
    plt.scatter(feature_values, y_test, color='blue', label='Actual')
    plt.scatter(feature_values, y_pred_svr, color='red', label='Predicted')
    
    plt.xlabel(feature)
    plt.ylabel('Log(Area)')
    plt.title(f'SVR: Actual vs. Predicted for {feature}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the estimated area for each feature based on the RF model
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features):
    # Create a subplot for each feature
    plt.subplot(4, 3, i+1)
    
    # Extract the current feature for plotting
    feature_values = [row[features.index(feature)] for row in X_test]
    
    # Plot the actual values against the predicted values
    plt.scatter(feature_values, y_test, color='blue', label='Actual')
    plt.scatter(feature_values, y_pred_rf, color='red', label='Predicted')
    
    plt.xlabel(feature)
    plt.ylabel('Log(Area)')
    plt.title(f'RF: Actual vs. Predicted for {feature}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


# Plot the estimated area for each feature based on the NN model
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features):
    # Create a subplot for each feature
    plt.subplot(4, 3, i+1)
    
    # Extract the current feature for plotting
    feature_values = [row[features.index(feature)] for row in X_test]
    
    # Plot the actual values against the predicted values
    plt.scatter(feature_values, y_test, color='blue', label='Actual')
    plt.scatter(feature_values, y_pred_nn, color='red', label='Predicted')
    
    plt.xlabel(feature)
    plt.ylabel('Log(Area)')
    plt.title(f'Neural Net: Actual vs. Predicted for {feature}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting Linear Regression feature coefficients
plt.bar(features + ['Intercept'], list(coefficients) + [bias_lr])
plt.title('Linear Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()

# Plot the training history for the neural network
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Neural Network Convergence')
plt.legend()
plt.grid(True)
plt.show()


# Get the support vectors and their indices
support_vectors = svr_model.support_vectors_
support_vectors_indices = svr_model.support_

# Identify the most important features based on the support vectors
weighted_features = np.sum(np.abs(support_vectors), axis=0)

# Normalize the importance values
feature_importance = weighted_features / np.sum(weighted_features)

# Plotting the bar graph
plt.bar(features, feature_importance)
plt.title('SVR Feature Importance Based on Support Vectors')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Plotting Random Forest feature importances
feature_importances = rf_model.feature_importances_

# Create a bar plot
plt.bar(features, feature_importances)
plt.title('Random Forest Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()


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