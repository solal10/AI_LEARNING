import numpy as np
import pandas as pd
import logging
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results[name] = {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "mse": mse,
        "r2": r2,
    }

    # Print results
    print(f"\n{name} Results:")
    print("Feature Coefficients:")
    for feature_name, coef in zip(housing.feature_names, model.coef_):
        print(f"{feature_name}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

# Visualize coefficients comparison
plt.figure(figsize=(15, 6))
x = np.arange(len(housing.feature_names))
width = 0.25

for i, (name, result) in enumerate(results.items()):
    plt.bar(x + i * width, result["coefficients"], width, label=name)

plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Comparison of Feature Coefficients Across Models")
plt.xticks(x + width, housing.feature_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Visualize actual vs predicted for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Actual vs Predicted House Prices")

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    axes[i].scatter(y_test, y_pred, alpha=0.5)
    axes[i].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
    )
    axes[i].set_xlabel("Actual Prices")
    axes[i].set_ylabel("Predicted Prices")
    axes[i].set_title(f'{name}\nR² = {results[name]["r2"]:.4f}')

plt.tight_layout()
plt.show()
