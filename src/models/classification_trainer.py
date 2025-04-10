import logging
import os
from typing import Dict, Tuple, List
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
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """Classe pour entraîner et évaluer des modèles de classification."""

    def __init__(
        self,
        model_type: str = "random_forest",
        hyperparams: Dict = None,
        cv_folds: int = 5,
        task: str = "classification"
    ):
        """
        Initialise le ClassificationTrainer.

        Args:
            model_type (str): Type de modèle à utiliser
            hyperparams (Dict): Hyperparamètres du modèle
            cv_folds (int): Nombre de folds pour la validation croisée
            task (str): Type de tâche (classification)
        """
        self.model_type = model_type
        self.hyperparams = hyperparams or {}
        self.cv_folds = cv_folds
        self.task = task
        self.model = None
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model(self) -> BaseEstimator:
        """Retourne le modèle approprié selon le type spécifié."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(**self.hyperparams)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(**self.hyperparams)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(**self.hyperparams)
        elif self.model_type == "knn":
            return KNeighborsClassifier(**self.hyperparams)
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(**self.hyperparams)
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")

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
        logger.info(f"Début de l'entraînement du modèle ({self.model_type})")

        # Créer le pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", self._get_model())
        ])

        # Entraîner le modèle
        pipeline.fit(X_train, y_train)
        self.model = pipeline

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
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculer les métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

        # Générer le rapport de classification
        report = classification_report(y_test, y_pred)
        logger.info("Métriques d'évaluation:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-score: {metrics['f1']:.4f}")
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
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            disp.plot()
            plt.title("Courbe ROC")
            plt.savefig(f"reports/roc_curve_{timestamp}.png")
            plt.close()

        # 3. Courbe Precision-Recall
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.title("Courbe Precision-Recall")
        plt.savefig(f"reports/precision_recall_curve_{timestamp}.png")
        plt.close()

    def save_model(self, model_name: str):
        """Sauvegarde le modèle."""
        model_path = os.path.join(self.model_dir, model_name)
        self.model.save(model_path)
        logger.info(f"Modèle sauvegardé dans {model_path}")

    def load_model(self, model_name: str):
        """Charge un modèle sauvegardé."""
        model_path = os.path.join(self.model_dir, model_name)
        self.model = Pipeline.load(model_path)
        logger.info(f"Modèle chargé depuis {model_path}")

    def compare_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, preprocessor: Pipeline
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare les performances de plusieurs modèles de classification.

        Args:
            X_train (pd.DataFrame): Features d'entraînement
            y_train (pd.Series): Target d'entraînement
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Target de test
            preprocessor (Pipeline): Preprocessor pour les transformations

        Returns:
            Dict[str, Dict[str, float]]: Métriques de performance pour chaque modèle
        """
        logger.info("Comparaison des performances des modèles...")

        # Définir les modèles à comparer
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42, enable_categorical=True)
        }

        # Initialiser le dictionnaire pour stocker les résultats
        results_dict = {}
        y_preds = {}

        # Évaluer chaque modèle
        for name, model in models.items():
            logger.info(f"Évaluation du modèle: {name}")
            
            # Créer le pipeline
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            # Entraîner le modèle
            pipeline.fit(X_train, y_train)

            # Faire les prédictions
            y_pred = pipeline.predict(X_test)
            y_preds[name] = y_pred

            # Calculer les métriques
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted')
            }

            # Ajouter les résultats dans le dictionnaire
            results_dict[name] = metrics

        # Créer le DataFrame à partir du dictionnaire
        results_df = pd.DataFrame.from_dict(results_dict, orient='index')
        results_df = results_df.reset_index().rename(columns={'index': 'Model'})
        
        # Trier par F1-score (meilleur modèle en premier)
        results_df = results_df.sort_values("f1", ascending=False)
        
        # Sauvegarder les résultats dans un CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.model_dir, f"model_comparison_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"Résultats de la comparaison sauvegardés dans {results_file}")
        logger.info("\nComparaison des modèles:")
        logger.info(results_df.to_string())

        return results_dict 