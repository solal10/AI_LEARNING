import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Classe pour générer les rapports et visualisations."""

    def __init__(self, output_dir="reports"):
        """
        Initialise le ReportGenerator.

        Args:
            output_dir (str): Dossier pour sauvegarder les rapports
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Créer le dossier reports s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_report(self, model_trainer, X_test, y_test, feature_names, metrics):
        """
        Génère le rapport complet avec toutes les visualisations.

        Args:
            model_trainer (ModelTrainer): Instance du ModelTrainer
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Valeurs réelles
            feature_names (list): Noms des features
            metrics (dict): Métriques du modèle
        """
        logger.info("Génération du rapport")

        # Obtenir les prédictions
        y_pred = model_trainer.model.predict(X_test)

        # Obtenir l'importance des features
        feature_importance = model_trainer.get_feature_importance(feature_names)

        # 1. Générer les visualisations
        self._plot_predictions_vs_real(y_test, y_pred)
        self._plot_feature_importance(feature_importance)
        self._plot_residuals(y_test, y_pred)

        # 2. Générer le rapport texte
        self._generate_text_report(metrics, feature_importance)

        logger.info("Rapport généré avec succès")

    def _plot_predictions_vs_real(self, y_test, y_pred):
        """
        Génère le graphique des prédictions vs valeurs réelles.

        Args:
            y_test (pd.Series): Valeurs réelles
            y_pred (np.array): Prédictions
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
        )
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Prédictions")
        plt.title("Valeurs réelles vs Prédictions")

        # Sauvegarder le graphique
        filename = f"predictions_vs_real_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()

        logger.info(f"Graphique sauvegardé: {filepath}")

    def _plot_feature_importance(self, feature_importance):
        """
        Génère le graphique de l'importance des features.

        Args:
            feature_importance (pd.DataFrame): Importance des features
        """
        # Déterminer le nom de la colonne d'importance
        importance_col = next(
            col for col in feature_importance.columns if col.startswith("Abs_")
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(x=importance_col, y="Feature", data=feature_importance)
        plt.title("Importance des features")
        plt.tight_layout()

        # Sauvegarder le graphique
        filename = f"feature_importance_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()

        logger.info(f"Graphique sauvegardé: {filepath}")

    def _plot_residuals(self, y_test, y_pred):
        """
        Génère le graphique des résidus.

        Args:
            y_test (pd.Series): Valeurs réelles
            y_pred (np.array): Prédictions
        """
        residuals = y_test - y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Prédictions")
        plt.ylabel("Résidus")
        plt.title("Graphique des résidus")

        # Sauvegarder le graphique
        filename = f"residuals_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()

        logger.info(f"Graphique sauvegardé: {filepath}")

    def _generate_text_report(self, model_metrics, feature_importance):
        """
        Génère le rapport texte avec les métriques et l'importance des features.

        Args:
            model_metrics (dict): Métriques du modèle
            feature_importance (pd.DataFrame): Importance des features
        """
        # Créer le contenu du rapport
        report_content = [
            "=== RAPPORT DE PERFORMANCE DU MODÈLE ===\n",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n=== MÉTRIQUES ===",
            f"RMSE: {model_metrics['rmse']:.4f}",
            f"MAE: {model_metrics['mae']:.4f}",
            f"R²: {model_metrics['r2']:.4f}",
            "\n=== IMPORTANCE DES FEATURES ===",
            feature_importance.to_string(),
        ]

        # Sauvegarder le rapport
        filename = f"model_report_{self.timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            f.write("\n".join(report_content))

        logger.info(f"Rapport sauvegardé: {filepath}")
