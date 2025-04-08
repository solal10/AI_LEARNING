import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from src.models.model_trainer import ModelTrainer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import os

@pytest.fixture
def sample_preprocessor():
    """Fixture qui fournit un preprocessor pour les tests."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['feature1', 'feature2'])
        ])

def test_init():
    """Test l'initialisation du ModelTrainer."""
    trainer = ModelTrainer(model_type='linear')
    assert trainer.model_type == 'linear'
    assert trainer.model is None
    assert isinstance(trainer.hyperparams, dict)

def test_get_model_linear():
    """Test la création d'un modèle linéaire."""
    trainer = ModelTrainer(model_type='linear')
    model = trainer._get_model()
    assert isinstance(model, LinearRegression)

def test_get_model_random_forest():
    """Test la création d'un modèle Random Forest."""
    hyperparams = {'n_estimators': 50, 'max_depth': 5}
    trainer = ModelTrainer(model_type='random_forest', hyperparams=hyperparams)
    model = trainer._get_model()
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 50
    assert model.max_depth == 5

def test_get_model_xgboost():
    """Test la création d'un modèle XGBoost."""
    hyperparams = {'n_estimators': 50, 'learning_rate': 0.1}
    trainer = ModelTrainer(model_type='xgboost', hyperparams=hyperparams)
    model = trainer._get_model()
    assert isinstance(model, xgb.XGBRegressor)
    assert model.n_estimators == 50
    assert model.learning_rate == 0.1

def test_get_model_invalid():
    """Test la gestion d'erreur pour un type de modèle invalide."""
    trainer = ModelTrainer(model_type='invalid_model')
    with pytest.raises(ValueError):
        trainer._get_model()

def test_train_model(sample_model_data, sample_preprocessor):
    """Test l'entraînement du modèle."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type='linear')
    
    model = trainer.train_model(X_train, y_train, sample_preprocessor)
    assert trainer.model is not None
    assert hasattr(model, 'predict')

def test_evaluate_model(sample_model_data, sample_preprocessor):
    """Test l'évaluation du modèle."""
    X_train, X_test, y_train, y_test = sample_model_data
    trainer = ModelTrainer(model_type='linear')
    
    # Entraîner le modèle
    trainer.train_model(X_train, y_train, sample_preprocessor)
    
    # Évaluer le modèle
    metrics = trainer.evaluate_model(X_test, y_test)
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_save_and_load_model(sample_model_data, sample_preprocessor, tmp_path):
    """Test la sauvegarde et le chargement du modèle."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type='linear', model_dir=str(tmp_path))
    
    # Entraîner et sauvegarder le modèle
    trainer.train_model(X_train, y_train, sample_preprocessor)
    trainer.save_model('test_model.joblib')
    
    # Vérifier que le fichier existe
    assert os.path.exists(os.path.join(str(tmp_path), 'test_model.joblib'))
    
    # Charger le modèle
    new_trainer = ModelTrainer(model_dir=str(tmp_path))
    new_trainer.load_model('test_model.joblib')
    
    assert new_trainer.model is not None

def test_get_feature_importance(sample_model_data, sample_preprocessor):
    """Test l'obtention de l'importance des features."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type='random_forest')
    
    # Entraîner le modèle
    trainer.train_model(X_train, y_train, sample_preprocessor)
    
    # Obtenir l'importance des features
    feature_names = ['feature1', 'feature2']
    importance_df = trainer.get_feature_importance(feature_names)
    
    assert len(importance_df) == len(feature_names)
    assert 'Feature' in importance_df.columns
    assert 'Importance' in importance_df.columns 