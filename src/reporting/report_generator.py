import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Classe pour générer les rapports et visualisations des modèles."""

    def __init__(self, output_dir="reports"):
        """
        Initialise le ReportGenerator.

        Args:
            output_dir (str): Dossier pour sauvegarder les rapports
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_residual_plots(
        self,
        y_true: np.ndarray,
        y_preds: Dict[str, np.ndarray],
        model_names: List[str],
        save_path: str = None,
    ):
        """
        Génère les plots de résidus pour chaque modèle.

        Args:
            y_true (np.ndarray): Valeurs réelles
            y_preds (Dict[str, np.ndarray]): Prédictions pour chaque modèle
            model_names (List[str]): Noms des modèles
            save_path (str): Chemin pour sauvegarder le plot
        """
        logger.info("Génération des plots de résidus...")

        # Calculer le nombre de lignes et colonnes pour la grille de plots
        n_models = len(model_names)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        # Créer la figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Générer un plot pour chaque modèle
        for i, (name, y_pred) in enumerate(y_preds.items()):
            # Calculer les résidus
            residuals = y_true - y_pred

            # Créer le plot
            ax = axes[i]
            sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_title(f'Résidus - {name}')
            ax.set_xlabel('Prédictions')
            ax.set_ylabel('Résidus')

        # Supprimer les axes vides
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        # Sauvegarder le plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"residual_plots_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Plots de résidus sauvegardés dans {save_path}")

    def generate_prediction_plots(
        self,
        y_true: np.ndarray,
        y_preds: Dict[str, np.ndarray],
        model_names: List[str],
        save_path: str = None,
    ):
        """
        Génère les plots de prédictions vs valeurs réelles pour chaque modèle.

        Args:
            y_true (np.ndarray): Valeurs réelles
            y_preds (Dict[str, np.ndarray]): Prédictions pour chaque modèle
            model_names (List[str]): Noms des modèles
            save_path (str): Chemin pour sauvegarder le plot
        """
        logger.info("Génération des plots de prédictions...")

        # Calculer le nombre de lignes et colonnes pour la grille de plots
        n_models = len(model_names)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        # Créer la figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Générer un plot pour chaque modèle
        for i, (name, y_pred) in enumerate(y_preds.items()):
            ax = axes[i]
            sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=ax)
            
            # Ajouter la ligne y=x
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_title(f'Prédictions vs Réelles - {name}')
            ax.set_xlabel('Valeurs réelles')
            ax.set_ylabel('Prédictions')

        # Supprimer les axes vides
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        # Sauvegarder le plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"prediction_plots_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Plots de prédictions sauvegardés dans {save_path}")

    def generate_comparison_report(
        self,
        y_true: np.ndarray,
        y_preds: Dict[str, np.ndarray],
        model_names: List[str],
        metrics: Dict[str, Dict[str, float]],
    ):
        """
        Génère un rapport complet de comparaison des modèles.

        Args:
            y_true (np.ndarray): Valeurs réelles
            y_preds (Dict[str, np.ndarray]): Prédictions pour chaque modèle
            model_names (List[str]): Noms des modèles
            metrics (Dict[str, Dict[str, float]]): Métriques pour chaque modèle
        """
        logger.info("Génération du rapport de comparaison...")

        # Générer les plots
        self.generate_residual_plots(y_true, y_preds, model_names)
        self.generate_prediction_plots(y_true, y_preds, model_names)

        # Créer un DataFrame avec les métriques
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
        
        # Sauvegarder les métriques
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(self.output_dir, f"model_metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"Métriques sauvegardées dans {metrics_file}") 