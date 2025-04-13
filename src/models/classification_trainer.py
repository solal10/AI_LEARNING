import logging
import os
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """Classe pour entraîner et évaluer des modèles de classification."""

    def __init__(self):
        """Initialize the classification trainer."""
        self.logger = logging.getLogger(__name__)
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'XGBoost': xgb.XGBClassifier(random_state=42)
        }
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model(self) -> BaseEstimator:
        """Retourne le modèle approprié selon le type spécifié."""
        return self.models['XGBoost']

    def train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, preprocessor: Pipeline
    ) -> Pipeline:
        """
        Entraîne le modèle de classification.

        Args:
            X_train (pd.DataFrame): Features d'entraînement
            y_train (pd.Series): Target d'entraînement
            preprocessor (Pipeline): Preprocessor pour les transformations

        Returns:
            Pipeline: Pipeline entraîné
        """
        logger.info("Début de l'entraînement du modèle (XGBoost)")

        # Créer le pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", self._get_model())
        ])

        # Entraîner le modèle
        pipeline.fit(X_train, y_train)

        logger.info("Modèle entraîné avec succès")
        return pipeline

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Évalue le modèle de classification.

        Args:
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Target de test

        Returns:
            Dict[str, float]: Métriques d'évaluation
        """
        logger.info("Évaluation du modèle")

        # Faire les prédictions
        y_pred = self.models['XGBoost'].predict(X_test)
        y_pred_proba = self.models['XGBoost'].predict_proba(X_test)[:, 1]

        # Calculer les métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=1),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        # Générer le rapport de classification
        report = classification_report(y_test, y_pred)
        logger.info("Métriques d'évaluation:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-score: {metrics['f1']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info("\nRapport de classification:")
        logger.info(report)

        # Générer les visualisations
        self._generate_classification_visualizations(y_test, y_pred, y_pred_proba)

        return metrics

    def _generate_classification_visualizations(
        self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ):
        """
        Génère les visualisations pour la classification.

        Args:
            y_true (pd.Series): Valeurs réelles
            y_pred (np.ndarray): Prédictions
            y_pred_proba (np.ndarray): Probabilités des prédictions
        """
        # Créer le dossier reports s'il n'existe pas
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Matrice de confusion
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matrice de confusion")
        plt.savefig(f"reports/confusion_matrix_{timestamp}.png")
        plt.close()

        # 2. Courbe ROC (si classification binaire)
        if len(np.unique(y_true)) == 2:
            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            disp.plot()
            plt.title("Courbe ROC")
            plt.savefig(f"reports/roc_curve_{timestamp}.png")
            plt.close()

        # 3. Courbe Precision-Recall
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.title("Courbe Precision-Recall")
        plt.savefig(f"reports/precision_recall_curve_{timestamp}.png")
        plt.close()

    def save_model(self, model_name: str):
        """Sauvegarde le modèle."""
        model_path = os.path.join(self.model_dir, model_name)
        import joblib
        joblib.dump(self.models['XGBoost'], model_path)
        logger.info(f"Modèle sauvegardé dans {model_path}")

    def load_model(self, model_name: str):
        """Charge un modèle sauvegardé."""
        model_path = os.path.join(self.model_dir, model_name)
        import joblib
        self.models['XGBoost'] = joblib.load(model_path)
        logger.info(f"Modèle chargé depuis {model_path}")

    def compare_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Compare multiple classification models and return their performance metrics."""
        try:
            results = {}
            best_accuracy = 0
            best_model = None
            
            for name, model in self.models.items():
                self.logger.info(f"Training and evaluating {name}...")
                
                # Create pipeline with scaler and model
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
                
                # Train the model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'roc_curve': (fpr, tpr, _),
                    'predictions': y_pred,
                    'model': pipeline
                }
                
                # Update best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = name
                    
            # Add best model information
            results['best_model'] = results[best_model]
            results['best_model_name'] = best_model
            
            self.logger.info(f"Best model: {best_model} with accuracy: {best_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model comparison: {str(e)}")
            raise 