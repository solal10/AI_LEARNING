import numpy as np
import os
from src.models.model_trainer import ModelTrainer
from sklearn.preprocessing import StandardScaler

def test_model_training():
    """Test l'entraînement du modèle."""
    # Create test data
    X = np.array([[50, 1, 140, 200, 0, 150, 0, 1.0],
                  [60, 0, 120, 180, 1, 140, 1, 2.0],
                  [70, 1, 130, 220, 0, 130, 0, 1.5]])
    y = np.array([0, 1, 0])

    # Initialize preprocessor
    preprocessor = StandardScaler()
    preprocessor.fit(X)

    # Initialize and train model
    model = ModelTrainer(model_type="random_forest", task="classification")
    model.train_model(X, y, preprocessor)

    # Make predictions
    predictions = model.predict(X)
    assert predictions.shape == (3,)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)

def test_model_prediction():
    """Test les prédictions du modèle."""
    # Create test data
    X = np.array([[50, 1, 140, 200, 0, 150, 0, 1.0],
                  [60, 0, 120, 180, 1, 140, 1, 2.0]])
    y = np.array([0, 1])

    # Initialize preprocessor
    preprocessor = StandardScaler()
    preprocessor.fit(X)

    # Train model
    model = ModelTrainer(model_type="random_forest", task="classification")
    model.train_model(X, y, preprocessor)

    # Test prediction
    X_test = np.array([[65, 1, 130, 190, 0, 145, 0, 1.5]])
    prediction = model.predict(X_test)
    assert isinstance(prediction[0], (np.int64, int))
    assert prediction.shape == (1,)

def test_model_save_load(tmp_path):
    """Test la sauvegarde et le chargement du modèle."""
    # Create test data
    X = np.array([[50, 1, 140, 200, 0, 150, 0, 1.0],
                  [60, 0, 120, 180, 1, 140, 1, 2.0]])
    y = np.array([0, 1])

    # Initialize preprocessor
    preprocessor = StandardScaler()
    preprocessor.fit(X)

    # Train and save model
    model = ModelTrainer(model_type="random_forest", task="classification")
    model.train_model(X, y, preprocessor)
    
    save_path = os.path.join(tmp_path, "model.joblib")
    model.save_model(save_path)
    assert os.path.exists(save_path)

    # Load model and make predictions
    loaded_model = ModelTrainer(model_type="random_forest", task="classification")
    loaded_model.load_model(save_path)
    predictions = loaded_model.predict(X)
    assert predictions.shape == (2,)