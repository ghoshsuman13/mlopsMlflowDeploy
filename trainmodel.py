import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("data/dataset.csv")
X = data[['feature']]
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name
mlflow.set_experiment("Simple Linear Regression Experiment")

# Loop over different noise levels for different runs
for noise in [0.1, 0.2, 0.3]:
    with mlflow.start_run():
        # Modify data (simulate noise)
        y_train_noisy = y_train + noise
        y_test_noisy = y_test + noise

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train_noisy)

        # Evaluate model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test_noisy, predictions)

        # Log parameters and metrics
        mlflow.log_param("noise", noise)
        mlflow.log_metric("mse", mse)

        # Log the model with input example
        input_example = X_test[:1]  # Take the first row of the test data as an input example
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        # Print and log the results
        print(f"Logged run with noise={noise}, MSE={mse}")
