import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_data():
    """Fixture qui fournit un DataFrame d'exemple pour les tests."""
    return pd.DataFrame(
        {
            "median_income": [8.3252, 8.3014, 7.2574],
            "housing_median_age": [41.0, 21.0, 52.0],
            "AveRooms": [6.984126984126984, 6.238137082601054, 8.288135593220339],
            "AveBedrms": [1.0238095238095237, 0.9718804920913884, 1.073446327683616],
            "population": [322.0, 2401.0, 496.0],
            "AveOccup": [2.5555555555555554, 2.109841827768014, 2.8022598870056497],
            "Latitude": [37.88, 37.86, 37.85],
            "Longitude": [-122.23, -122.22, -122.24],
            "median_house_value": [452600.0, 358500.0, 352100.0],
            "ocean_proximity": ["NEAR OCEAN", "ISLAND", "INLAND"],
        }
    )


@pytest.fixture
def sample_preprocessed_data(sample_data):
    """Fixture qui fournit des données prétraitées pour les tests."""
    X = sample_data.drop("median_house_value", axis=1)
    y = sample_data["median_house_value"]
    return X, y


@pytest.fixture
def sample_model_data():
    """Fixture qui fournit des données d'entraînement et de test pour les modèles."""
    np.random.seed(42)
    X_train = pd.DataFrame(
        {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
    )
    y_train = np.random.randn(100)
    X_test = pd.DataFrame(
        {"feature1": np.random.randn(20), "feature2": np.random.randn(20)}
    )
    y_test = np.random.randn(20)
    return X_train, X_test, y_train, y_test
