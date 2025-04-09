import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from datetime import datetime
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Classe pour entraîner et évaluer le modèle."""
    
    def __init__(self, model_dir='models', model_type='linear', hyperparams=None, cv_folds=5):
        """
        Initialise le ModelTrainer.
        
        Args:
            model_dir (str): Dossier pour sauvegarder les modèles
            model_type (str): Type de modèle ('linear', 'random_forest', 'xgboost')
            hyperparams (dict): Hyperparamètres pour le modèle
            cv_folds (int): Nombre de folds pour la validation croisée
        """
        self.model_dir = model_dir
        self.model = None
        self.metrics = {}
        self.model_type = model_type
        self.hyperparams = hyperparams or {}
        self.cv_folds = cv_folds
        
        # Créer le dossier models s'il n'existe pas
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def _get_model(self):
        """
        Retourne l'instance du modèle selon le type choisi.
        
        Returns:
            object: Instance du modèle
        """
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', None),
                min_samples_split=self.hyperparams.get('min_samples_split', 2),
                min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
                random_state=self.hyperparams.get('random_state', 42)
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', 6),
                learning_rate=self.hyperparams.get('learning_rate', 0.1),
                subsample=self.hyperparams.get('subsample', 1.0),
                colsample_bytree=self.hyperparams.get('colsample_bytree', 1.0),
                random_state=self.hyperparams.get('random_state', 42),
                enable_categorical=True
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")

    def train_model(self, X_train, y_train, preprocessor, tune_hyperparams=False):
        """
        Entraîne le modèle avec ou sans recherche d'hyperparamètres.
        
        Args:
            X_train (pd.DataFrame): Features d'entraînement
            y_train (pd.Series): Target d'entraînement
            preprocessor (ColumnTransformer): Preprocessor pour les transformations
            tune_hyperparams (bool): Si True, effectue une recherche d'hyperparamètres
            
        Returns:
            Pipeline: Modèle entraîné
        """
        logger.info(f"Début de l'entraînement du modèle ({self.model_type})")
        
        # Créer le pipeline complet
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self._get_model())
        ])
        
        if tune_hyperparams:
            # Effectuer la recherche d'hyperparamètres
            self.tune_hyperparameters(X_train, y_train)
        else:
            # Entraîner le modèle directement
            self.model.fit(X_train, y_train)
        
        logger.info("Modèle entraîné avec succès")
        return self.model
    
    def _tune_hyperparameters(self, X_train, y_train):
        """
        Effectue une recherche d'hyperparamètres par GridSearchCV.
        
        Args:
            X_train (pd.DataFrame): Features d'entraînement
            y_train (pd.Series): Target d'entraînement
        """
        logger.info("Début de la recherche d'hyperparamètres")
        
        # Définir les grilles de paramètres selon le type de modèle
        if self.model_type == 'random_forest':
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'xgboost':
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [3, 6, 9],
                'regressor__learning_rate': [0.01, 0.1, 0.3],
                'regressor__subsample': [0.8, 0.9, 1.0],
                'regressor__colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            logger.info("Pas de recherche d'hyperparamètres pour ce type de modèle")
            return
        
        # Créer le GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Effectuer la recherche
        grid_search.fit(X_train, y_train)
        
        # Mettre à jour le modèle avec les meilleurs paramètres
        self.model = grid_search.best_estimator_
        logger.info(f"Meilleurs paramètres trouvés: {grid_search.best_params_}")
        logger.info(f"Meilleur score: {-grid_search.best_score_:.4f} (RMSE)")
        
        # Mettre à jour les hyperparamètres avec les meilleurs trouvés
        for param, value in grid_search.best_params_.items():
            param_name = param.replace('regressor__', '')
            self.hyperparams[param_name] = value

    def evaluate_model(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test.
        
        Args:
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Target de test
            
        Returns:
            dict: Dictionnaire contenant les métriques d'évaluation
        """
        logger.info("Évaluation du modèle")
        
        # Faire les prédictions
        y_pred = self.model.predict(X_test)
        
        # Calculer les métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Stocker les métriques
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Afficher les métriques
        logger.info("Métriques d'évaluation:")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R2: {r2:.4f}")
        
        return self.metrics
    
    def save_model(self, filename='model.joblib'):
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            filename (str): Nom du fichier pour sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train_model d'abord.")
        
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, filepath)
        logger.info(f"Modèle sauvegardé dans {filepath}")
    
    def load_model(self, filename='model.joblib'):
        """
        Charge un modèle sauvegardé.
        
        Args:
            filename (str): Nom du fichier du modèle à charger
        """
        filepath = os.path.join(self.model_dir, filename)
        self.model = joblib.load(filepath)
        logger.info(f"Modèle chargé depuis {filepath}")
    
    def get_feature_importance(self, feature_names):
        """
        Obtient l'importance des features du modèle.
        
        Args:
            feature_names (list): Liste des noms des features
            
        Returns:
            pd.DataFrame: DataFrame avec l'importance des features
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train_model d'abord.")
        
        regressor = self.model.named_steps['regressor']
        
        if isinstance(regressor, LinearRegression):
            # Pour la régression linéaire, utiliser les coefficients
            importance = regressor.coef_
            importance_name = 'Coefficient'
        elif isinstance(regressor, (RandomForestRegressor, xgb.XGBRegressor)):
            # Pour Random Forest et XGBoost, utiliser feature_importances_
            importance = regressor.feature_importances_
            importance_name = 'Importance'
        else:
            raise ValueError(f"Calcul d'importance non supporté pour le modèle {type(regressor)}")
        
        # Créer un DataFrame avec les importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            importance_name: importance
        })
        
        # Trier par importance absolue
        importance_df[f'Abs_{importance_name}'] = importance_df[importance_name].abs()
        importance_df = importance_df.sort_values(f'Abs_{importance_name}', ascending=False)
        
        return importance_df 

    def cross_validate_model(self, X, y, save_scores=True):
        """
        Effectue une validation croisée sur le modèle.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            save_scores (bool): Si True, sauvegarde les scores dans un fichier CSV
            
        Returns:
            dict: Dictionnaire contenant les scores moyens et leurs écarts-types
        """
        logger.info(f"Début de la validation croisée avec {self.cv_folds} folds")
        
        # Définir les métriques à calculer
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        
        # Effectuer la validation croisée
        cv_results = cross_validate(
            estimator=self.model,
            X=X,
            y=y,
            cv=self.cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculer les moyennes et écarts-types
        cv_metrics = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            # Inverser le signe pour les métriques négatives
            if metric in ['rmse', 'mae']:
                test_scores = -test_scores
                train_scores = -train_scores
            
            cv_metrics[f'cv_{metric}_mean'] = np.mean(test_scores)
            cv_metrics[f'cv_{metric}_std'] = np.std(test_scores)
            cv_metrics[f'cv_{metric}_train_mean'] = np.mean(train_scores)
            cv_metrics[f'cv_{metric}_train_std'] = np.std(train_scores)
        
        # Afficher les résultats
        logger.info("Résultats de la validation croisée:")
        for metric in scoring.keys():
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Test - Moyenne: {cv_metrics[f'cv_{metric}_mean']:.4f} ± {cv_metrics[f'cv_{metric}_std']:.4f}")
            logger.info(f"  Train - Moyenne: {cv_metrics[f'cv_{metric}_train_mean']:.4f} ± {cv_metrics[f'cv_{metric}_train_std']:.4f}")
        
        # Sauvegarder les scores si demandé
        if save_scores:
            self._save_cv_scores(cv_results)
        
        return cv_metrics
    
    def _save_cv_scores(self, cv_results):
        """
        Sauvegarde les scores de validation croisée dans un fichier CSV.
        
        Args:
            cv_results (dict): Résultats de la validation croisée
        """
        # Créer un DataFrame avec les scores
        scores_df = pd.DataFrame({
            'fold': range(1, self.cv_folds + 1),
            'test_rmse': -cv_results['test_rmse'],
            'test_mae': -cv_results['test_mae'],
            'test_r2': cv_results['test_r2'],
            'train_rmse': -cv_results['train_rmse'],
            'train_mae': -cv_results['train_mae'],
            'train_r2': cv_results['train_r2']
        })
        
        # Créer le nom du fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'cv_scores_{self.model_type}_{timestamp}.csv'
        filepath = os.path.join(self.model_dir, filename)
        
        # Sauvegarder le DataFrame
        scores_df.to_csv(filepath, index=False)
        logger.info(f"Scores de validation croisée sauvegardés dans {filepath}")

    def save_results(self, model, test_metrics, cv_metrics):
        """
        Sauvegarde les résultats du modèle et les métriques.
        
        Args:
            model: Modèle entraîné
            test_metrics (dict): Métriques sur l'ensemble de test
            cv_metrics (dict): Métriques de validation croisée
        """
        # Créer un DataFrame avec toutes les métriques
        results = {
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'cv_rmse_mean': cv_metrics['cv_rmse_mean'],
            'cv_rmse_std': cv_metrics['cv_rmse_std'],
            'cv_mae_mean': cv_metrics['cv_mae_mean'],
            'cv_mae_std': cv_metrics['cv_mae_std'],
            'cv_r2_mean': cv_metrics['cv_r2_mean'],
            'cv_r2_std': cv_metrics['cv_r2_std']
        }
        
        # Sauvegarder les métriques dans un fichier CSV
        results_df = pd.DataFrame([results])
        results_file = os.path.join(self.model_dir, 'model_results.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"Résultats sauvegardés dans {results_file}")
        
        # Sauvegarder le modèle
        self.save_model('final_model')

    def tune_hyperparameters(self, X, y, param_grid=None, scoring='neg_root_mean_squared_error'):
        """
        Effectue une recherche d'hyperparamètres avec GridSearchCV.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            param_grid (dict): Grille de paramètres à explorer
            scoring (str): Métrique d'évaluation
            
        Returns:
            dict: Meilleurs paramètres trouvés
        """
        logger.info("Début de la recherche d'hyperparamètres")
        
        # Définir la grille de paramètres par défaut selon le type de modèle
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [None, 10, 20, 30],
                    'regressor__min_samples_split': [2, 5, 10],
                    'regressor__min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'xgboost':
                param_grid = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [3, 6, 9],
                    'regressor__learning_rate': [0.01, 0.1, 0.3],
                    'regressor__subsample': [0.8, 0.9, 1.0],
                    'regressor__colsample_bytree': [0.8, 0.9, 1.0]
                }
            elif self.model_type == 'linear':
                param_grid = {
                    'regressor__fit_intercept': [True, False],
                    'regressor__normalize': [True, False]
                }
        
        # Créer le GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Effectuer la recherche
        logger.info("Recherche des meilleurs hyperparamètres en cours...")
        grid_search.fit(X, y)
        
        # Mettre à jour le modèle avec les meilleurs paramètres
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # Afficher les résultats
        logger.info(f"Meilleurs paramètres trouvés: {self.best_params}")
        logger.info(f"Meilleur score: {self.best_score:.4f}")
        
        # Sauvegarder les résultats
        self._save_tuning_results(grid_search)
        
        return self.best_params
    
    def _save_tuning_results(self, grid_search):
        """
        Sauvegarde les résultats de la recherche d'hyperparamètres.
        
        Args:
            grid_search (GridSearchCV): Objet GridSearchCV après la recherche
        """
        # Créer un DataFrame avec les résultats
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Sélectionner les colonnes importantes
        important_columns = [
            'params',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
        
        # Ajouter les colonnes d'entraînement si elles existent
        if 'mean_train_score' in results_df.columns:
            important_columns.extend(['mean_train_score', 'std_train_score'])
        
        results_df = results_df[important_columns]
        
        # Créer le nom du fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'hyperparameter_tuning_{self.model_type}_{timestamp}.csv'
        filepath = os.path.join(self.model_dir, filename)
        
        # Sauvegarder le DataFrame
        results_df.to_csv(filepath, index=False)
        logger.info(f"Résultats de la recherche d'hyperparamètres sauvegardés dans {filepath}")
        
        # Sauvegarder les meilleurs paramètres dans un fichier JSON
        best_params_file = os.path.join(self.model_dir, f'best_params_{self.model_type}_{timestamp}.json')
        with open(best_params_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        logger.info(f"Meilleurs paramètres sauvegardés dans {best_params_file}") 