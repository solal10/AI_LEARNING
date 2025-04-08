import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Classe pour entraîner et évaluer le modèle."""
    
    def __init__(self, model_dir='models', model_type='linear', hyperparams=None):
        """
        Initialise le ModelTrainer.
        
        Args:
            model_dir (str): Dossier pour sauvegarder les modèles
            model_type (str): Type de modèle ('linear', 'random_forest', 'xgboost')
            hyperparams (dict): Hyperparamètres pour le modèle
        """
        self.model_dir = model_dir
        self.model = None
        self.metrics = {}
        self.model_type = model_type
        self.hyperparams = hyperparams or {}
        
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
        Entraîne le modèle.
        
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
            self._tune_hyperparameters(X_train, y_train)
        
        # Entraîner le modèle
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