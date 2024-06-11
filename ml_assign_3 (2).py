import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set the matplotlib backend for Mac
import matplotlib
matplotlib.use('MacOSX')

print("******************************* Beginning Of TASK-1 Regression Model *******************************")

# Load the regression dataset
cpu = fetch_openml(data_id=43946)
cpu.data = cpu.data.astype(np.float32)  # Convert features to float
cpu.target = cpu.target.astype(np.float32)  # Convert target to float

# Split data into train, validation, and test sets for regression
x_temp, x_test, y_temp, y_test = train_test_split(cpu.data, cpu.target, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=0)

# Initialize an empty DataFrame to store the minimum validation errors for regression
results_regression = pd.DataFrame(columns=['Model', 'Min Validation MSE'])

# Model configurations for regression
configs_regression = [
    {'layers': [64, 32], 'activation': 'relu'},
    {'layers': [100, 50, 25], 'activation': 'relu'},
    {'layers': [30, 30, 30], 'activation': 'sigmoid'},
    {'layers': [50, 25, 10, 5], 'activation': 'tanh'}
]

# Build and train different models for regression
for i, config in enumerate(configs_regression):
    model = Sequential([Dense(units, activation=config['activation']) for units in config['layers']])
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=600, validation_data=(x_val, y_val), verbose=0)

    # Plotting the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title(f'Regression Model {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

    # Storing results
    min_val_mse = min(history.history['val_mse'])
    new_row = pd.DataFrame({'Model': [f'Model {i+1}'], 'Min Validation MSE': [min_val_mse]})
    results_regression = pd.concat([results_regression, new_row], ignore_index=True)

# Print the regression results table
print("Regression Results:")
print(results_regression)
print("\n******************************* End Of TASK-1 *******************************")

print("\n******************************* Beginning of TASK-2 Classification Model *******************************")

# Load the classification dataset
dia = fetch_openml(data_id=41964)
dia.data = dia.data.astype(np.float32)
dia.target = dia.target.astype(np.int32)  # Ensure target is integer for classification

# Split data into train, validation, and test sets for classification
x_temp, x_test, y_temp, y_test = train_test_split(dia.data, dia.target, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=0)

# Initialize an empty DataFrame to store the maximum validation accuracies for classification
results_classification = pd.DataFrame(columns=['Model', 'Max Validation Accuracy'])

# Model configurations for classification
configs_classification = [
    {'layers': [64, 32], 'activation': 'relu'},
    {'layers': [100, 50, 25], 'activation': 'relu'},
    {'layers': [30, 30, 30], 'activation': 'sigmoid'},
    {'layers': [50, 25, 10, 5], 'activation': 'tanh'}
]

# Build and train different models for classification
for i, config in enumerate(configs_classification):
    model = Sequential([Dense(units, activation=config['activation']) for units in config['layers']])
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), verbose=0)

    # Plotting the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Classification Model {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Storing results
    max_val_accuracy = max(history.history['val_accuracy'])
    new_row = pd.DataFrame({'Model': [f'Model {i+1}'], 'Max Validation Accuracy': [max_val_accuracy]})
    results_classification = pd.concat([results_classification, new_row], ignore_index=True)

# Print the classification results table
print("Classification Results:")
print(results_classification)
print("\n******************************* End Of TASK-2 *******************************")