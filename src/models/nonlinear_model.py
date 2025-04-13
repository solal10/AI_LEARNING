import numpy as np
import pandas as pd
import logging
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
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
    results[name] = {"mse": mse, "r2": r2}

    # Print results
    print(f"\n{name} Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    # Feature importance for tree-based models
    if hasattr(model, "feature_importances_"):
        print("\nFeature Importances:")
        for feature_name, importance in zip(
            housing.feature_names, model.feature_importances_
        ):
            print(f"{feature_name}: {importance:.4f}")

# Visualize feature importance comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(housing.feature_names))
width = 0.35

for i, (name, model) in enumerate(models.items()):
    if hasattr(model, "feature_importances_"):
        plt.bar(x + i * width, model.feature_importances_, width, label=name)

plt.xlabel("Features")
plt.ylabel("Feature Importance")
plt.title("Comparison of Feature Importance Across Models")
plt.xticks(x + width / 2, housing.feature_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Visualize actual vs predicted for each model
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
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

# Compare with linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

linear_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
}

linear_results = {}

# Train and evaluate linear models
for name, model in linear_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    linear_results[name] = {"mse": mse, "r2": r2}

# Compare R² scores
plt.figure(figsize=(10, 6))
models_comparison = {**linear_results, **results}
model_names = list(models_comparison.keys())
r2_scores = [models_comparison[name]["r2"] for name in model_names]

plt.bar(model_names, r2_scores)
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("Comparison of R² Scores Across All Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
