import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from src.models.model_trainer import ModelTrainer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime


@pytest.fixture
def sample_preprocessor():
    """Fixture qui fournit un preprocessor pour les tests."""
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), ["feature1", "feature2"])]
    )


def test_init():
    """Test l'initialisation du ModelTrainer."""
    trainer = ModelTrainer(model_type="linear")
    assert trainer.model_type == "linear"
    assert trainer.model is None
    assert isinstance(trainer.hyperparams, dict)


def test_get_model_linear():
    """Test la création d'un modèle linéaire."""
    trainer = ModelTrainer(model_type="linear")
    model = trainer._get_model()
    assert isinstance(model, LinearRegression)


def test_get_model_random_forest():
    """Test la création d'un modèle Random Forest."""
    hyperparams = {"n_estimators": 50, "max_depth": 5}
    trainer = ModelTrainer(model_type="random_forest", hyperparams=hyperparams)
    model = trainer._get_model()
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 50
    assert model.max_depth == 5


def test_get_model_xgboost():
    """Test la création d'un modèle XGBoost."""
    hyperparams = {"n_estimators": 50, "learning_rate": 0.1}
    trainer = ModelTrainer(model_type="xgboost", hyperparams=hyperparams)
    model = trainer._get_model()
    assert isinstance(model, xgb.XGBRegressor)
    assert model.n_estimators == 50
    assert model.learning_rate == 0.1


def test_get_model_invalid():
    """Test la gestion d'erreur pour un type de modèle invalide."""
    trainer = ModelTrainer(model_type="invalid_model")
    with pytest.raises(ValueError):
        trainer._get_model()


def test_train_model(sample_model_data, sample_preprocessor):
    """Test l'entraînement du modèle."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type="linear")

    model = trainer.train_model(X_train, y_train, sample_preprocessor)
    assert trainer.model is not None
    assert hasattr(model, "predict")


def test_evaluate_model(sample_model_data, sample_preprocessor):
    """Test l'évaluation du modèle."""
    X_train, X_test, y_train, y_test = sample_model_data
    trainer = ModelTrainer(model_type="linear")

    # Entraîner le modèle
    trainer.train_model(X_train, y_train, sample_preprocessor)

    # Évaluer le modèle
    metrics = trainer.evaluate_model(X_test, y_test)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_save_and_load_model(sample_model_data, sample_preprocessor, tmp_path):
    """Test la sauvegarde et le chargement du modèle."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type="linear", model_dir=str(tmp_path))

    # Entraîner et sauvegarder le modèle
    trainer.train_model(X_train, y_train, sample_preprocessor)
    trainer.save_model("test_model.joblib")

    # Vérifier que le fichier existe
    assert os.path.exists(os.path.join(str(tmp_path), "test_model.joblib"))

    # Charger le modèle
    new_trainer = ModelTrainer(model_dir=str(tmp_path))
    new_trainer.load_model("test_model.joblib")

    assert new_trainer.model is not None


def test_get_feature_importance(sample_model_data, sample_preprocessor):
    """Test l'obtention de l'importance des features."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type="random_forest")

    # Entraîner le modèle
    trainer.train_model(X_train, y_train, sample_preprocessor)

    # Obtenir l'importance des features
    feature_names = ["feature1", "feature2"]
    importance_df = trainer.get_feature_importance(feature_names)

    assert len(importance_df) == len(feature_names)
    assert "Feature" in importance_df.columns
    assert "Importance" in importance_df.columns


def test_compare_models(sample_model_data, sample_preprocessor):
    """Test la comparaison des différents modèles."""
    X_train, X_test, y_train, y_test = sample_model_data
    trainer = ModelTrainer(model_type="linear", task="regression")
    
    # Comparer les modèles
    results = trainer.compare_models(X_train, y_train, X_test, y_test, sample_preprocessor)
    
    # Vérifier que tous les modèles sont présents
    expected_models = ["Linear Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]
    assert all(model in results for model in expected_models)
    
    # Vérifier que chaque modèle a les métriques requises
    for model_name, metrics in results.items():
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "R2" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


def test_cross_validation(sample_model_data, sample_preprocessor):
    """Test la validation croisée."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type="random_forest", cv_folds=3)
    
    # Entraîner le modèle
    trainer.train_model(X_train, y_train, sample_preprocessor)
    
    # Effectuer la validation croisée
    cv_metrics = trainer.cross_validate_model(X_train, y_train)
    
    # Vérifier les métriques
    assert "cv_rmse_mean" in cv_metrics
    assert "cv_rmse_std" in cv_metrics
    assert "cv_mae_mean" in cv_metrics
    assert "cv_mae_std" in cv_metrics
    assert "cv_r2_mean" in cv_metrics
    assert "cv_r2_std" in cv_metrics


def test_hyperparameter_tuning(sample_model_data, sample_preprocessor):
    """Test la recherche d'hyperparamètres."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(model_type="random_forest", should_tune_hyperparameters=True)
    
    # Entraîner le modèle avec tuning
    model = trainer.train_model(X_train, y_train, sample_preprocessor)
    
    # Vérifier que le modèle a été entraîné
    assert model is not None
    assert hasattr(model, "predict")


def test_save_comparison_results(sample_model_data, sample_preprocessor, tmp_path):
    """Test la sauvegarde des résultats de comparaison."""
    X_train, X_test, y_train, y_test = sample_model_data
    trainer = ModelTrainer(model_type="linear", model_dir=str(tmp_path))
    
    # Comparer les modèles
    trainer.compare_models(X_train, y_train, X_test, y_test, sample_preprocessor)
    
    # Vérifier que le fichier de résultats a été créé
    files = os.listdir(str(tmp_path))
    comparison_files = [f for f in files if f.startswith("model_comparison_")]
    assert len(comparison_files) > 0
    
    # Vérifier le contenu du fichier
    latest_file = max(comparison_files, key=lambda x: datetime.strptime(x.split("_")[2], "%Y%m%d"))
    df = pd.read_csv(os.path.join(str(tmp_path), latest_file))
    assert "Model" in df.columns
    assert "RMSE" in df.columns
    assert "MAE" in df.columns
    assert "R2" in df.columns


def test_task_handling(sample_model_data, sample_preprocessor):
    """Test la gestion des différents types de tâches."""
    X_train, X_test, y_train, y_test = sample_model_data
    
    # Test pour la régression
    regression_trainer = ModelTrainer(model_type="random_forest", task="regression")
    regression_trainer.train_model(X_train, y_train, sample_preprocessor)
    regression_metrics = regression_trainer.evaluate_model(X_test, y_test)
    assert all(metric in regression_metrics for metric in ["rmse", "mae", "r2"])
    
    # Test pour la classification
    classification_trainer = ModelTrainer(model_type="random_forest", task="classification")
    # Convertir y en classes pour la classification
    y_train_class = pd.qcut(y_train, q=3, labels=[0, 1, 2])
    y_test_class = pd.qcut(y_test, q=3, labels=[0, 1, 2])
    
    # Entraîner le modèle de classification
    classification_trainer.train_model(X_train, y_train_class, sample_preprocessor)
    
    # Évaluer le modèle
    classification_metrics = classification_trainer.evaluate_model(X_test, y_test_class)
    assert all(metric in classification_metrics for metric in ["accuracy", "precision", "recall", "f1"])


def test_error_handling():
    """Test la gestion des erreurs."""
    trainer = ModelTrainer(model_type="linear")
    
    # Test de l'erreur lors de la sauvegarde d'un modèle non entraîné
    with pytest.raises(ValueError, match="Le modèle n'a pas été entraîné"):
        trainer.save_model()
    
    # Test de l'erreur lors de l'obtention de l'importance des features sans modèle
    with pytest.raises(ValueError, match="Le modèle n'a pas été entraîné"):
        trainer.get_feature_importance(["feature1", "feature2"])
    
    # Test de l'erreur avec un type de modèle invalide
    with pytest.raises(ValueError, match="Type de modèle non supporté"):
        invalid_trainer = ModelTrainer(model_type="invalid_model")
        invalid_trainer._get_model()


def test_classification_models(sample_model_data, sample_preprocessor):
    """Test les différents types de modèles en classification."""
    X_train, X_test, y_train, y_test = sample_model_data
    
    # Convertir y en classes pour la classification
    y_train_class = pd.qcut(y_train, q=3, labels=[0, 1, 2])
    y_test_class = pd.qcut(y_test, q=3, labels=[0, 1, 2])
    
    # Tester Random Forest Classifier
    rf_trainer = ModelTrainer(model_type="random_forest", task="classification")
    rf_trainer.train_model(X_train, y_train_class, sample_preprocessor)
    rf_metrics = rf_trainer.evaluate_model(X_test, y_test_class)
    assert all(metric in rf_metrics for metric in ["accuracy", "precision", "recall", "f1"])
    
    # Tester XGBoost Classifier
    xgb_trainer = ModelTrainer(model_type="xgboost", task="classification")
    xgb_trainer.train_model(X_train, y_train_class, sample_preprocessor)
    xgb_metrics = xgb_trainer.evaluate_model(X_test, y_test_class)
    assert all(metric in xgb_metrics for metric in ["accuracy", "precision", "recall", "f1"])


def test_save_tuning_results(sample_model_data, sample_preprocessor, tmp_path):
    """Test la sauvegarde des résultats de tuning."""
    X_train, _, y_train, _ = sample_model_data
    trainer = ModelTrainer(
        model_type="random_forest",
        model_dir=str(tmp_path),
        should_tune_hyperparameters=True
    )
    
    # Entraîner le modèle avec tuning
    trainer.train_model(X_train, y_train, sample_preprocessor)
    
    # Vérifier que les fichiers de résultats ont été créés
    files = os.listdir(str(tmp_path))
    tuning_files = [f for f in files if f.startswith("tuning_results_")]
    best_params_files = [f for f in files if f.startswith("best_params_")]
    
    assert len(tuning_files) > 0
    assert len(best_params_files) > 0
    
    # Vérifier le contenu du fichier de tuning
    latest_tuning_file = max(tuning_files)
    df = pd.read_csv(os.path.join(str(tmp_path), latest_tuning_file))
    assert "mean_test_score" in df.columns
    assert "std_test_score" in df.columns
    assert "rank_test_score" in df.columns


def test_cross_validation_classification(sample_model_data, sample_preprocessor):
    """Test la validation croisée pour la classification."""
    X_train, _, y_train, _ = sample_model_data
    
    # Convertir y en classes pour la classification
    y_train_class = pd.qcut(y_train, q=3, labels=[0, 1, 2])
    
    # Créer un trainer pour la classification
    trainer = ModelTrainer(
        model_type="random_forest",
        task="classification",
        cv_folds=3
    )
    
    # Entraîner le modèle
    trainer.train_model(X_train, y_train_class, sample_preprocessor)
    
    # Effectuer la validation croisée
    cv_metrics = trainer.cross_validate_model(X_train, y_train_class)
    
    # Vérifier les métriques de classification
    assert "cv_accuracy_mean" in cv_metrics
    assert "cv_accuracy_std" in cv_metrics
    assert "cv_precision_mean" in cv_metrics
    assert "cv_precision_std" in cv_metrics
    assert "cv_recall_mean" in cv_metrics
    assert "cv_recall_std" in cv_metrics
    assert "cv_f1_mean" in cv_metrics
    assert "cv_f1_std" in cv_metrics


def test_model_comparison_with_custom_params(sample_model_data, sample_preprocessor):
    """Test la comparaison des modèles avec des paramètres personnalisés."""
    X_train, X_test, y_train, y_test = sample_model_data
    
    # Créer un trainer avec des hyperparamètres personnalisés
    custom_params = {
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 3
    }
    
    trainer = ModelTrainer(
        model_type="random_forest",
        hyperparams=custom_params,
        task="regression"
    )
    
    # Comparer les modèles
    results = trainer.compare_models(X_train, y_train, X_test, y_test, sample_preprocessor)
    
    # Vérifier que tous les modèles sont présents avec les bonnes métriques
    assert "Random Forest" in results
    assert all(metric in results["Random Forest"] for metric in ["RMSE", "MAE", "R2"])
    
    # Vérifier que les hyperparamètres personnalisés ont été utilisés
    rf_model = trainer._get_model()
    assert rf_model.n_estimators == custom_params["n_estimators"]
    assert rf_model.max_depth == custom_params["max_depth"]
    assert rf_model.min_samples_split == custom_params["min_samples_split"]
