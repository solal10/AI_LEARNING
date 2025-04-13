import pandas as pd
import numpy as np
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Classe pour prétraiter les données du California Housing Dataset.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, task: str = "regression"):
        """
        Initialise le DataPreprocessor.

        Args:
            test_size (float): Proportion du dataset à utiliser pour le test
            random_state (int): Seed pour la reproductibilité
            task (str): Type de tâche (regression ou classification)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.task = task
        self.preprocessor = None
        self.numerical_features = None
        self.categorical_features = None

    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """
        Prépare les données pour l'entraînement.

        Args:
            data (pd.DataFrame): Données à préparer

        Returns:
            tuple: X_train, X_test, y_train, y_test, preprocessor
        """
        if data.empty:
            raise ValueError("Le DataFrame est vide")

        # Séparer les features et la target
        X = data.drop("median_house_value", axis=1)
        y = data["median_house_value"]

        # Transformer la target en classes si classification
        if self.task == "classification":
            y = self._convert_to_classes(y)

        # Identifier les colonnes numériques et catégorielles
        self.numerical_features = X.select_dtypes(include=[np.number]).columns
        self.categorical_features = X.select_dtypes(include=["object", "category"]).columns

        logger.info(f"Features numériques: {len(self.numerical_features)}")
        logger.info(f"Features catégorielles: {len(self.categorical_features)}")

        # Créer le preprocessor
        self.preprocessor = self._create_preprocessor()

        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Taille du dataset d'entraînement: {X_train.shape}")
        logger.info(f"Taille du dataset de test: {X_test.shape}")

        return X_train, X_test, y_train, y_test, self.preprocessor

    def _convert_to_classes(self, y: pd.Series) -> pd.Series:
        """
        Convertit la variable cible en classes pour la classification.

        Args:
            y (pd.Series): Variable cible continue

        Returns:
            pd.Series: Variable cible discrétisée
        """
        # Utiliser des quantiles pour créer 5 classes
        y_class = pd.qcut(y, q=5, labels=[0, 1, 2, 3, 4])
        logger.info("Distribution des classes:")
        logger.info(y_class.value_counts(normalize=True))
        return y_class

    def _create_preprocessor(self, numeric_features=None, categorical_features=None) -> Pipeline:
        """
        Crée le preprocessor pour les transformations.

        Args:
            numeric_features (list): Liste des features numériques
            categorical_features (list): Liste des features catégorielles

        Returns:
            Pipeline: Preprocessor pour les transformations
        """
        # Utiliser les features fournies ou celles détectées
        numeric_features = numeric_features or self.numerical_features
        categorical_features = categorical_features or self.categorical_features

        # Définir les transformations
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # Créer le preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return preprocessor

    def get_feature_names(self, X):
        """
        Retourne les noms des features après transformation.

        Args:
            X (pd.DataFrame): DataFrame contenant les features

        Returns:
            list: Liste des noms de features après transformation
        """
        if not hasattr(self.preprocessor, "named_transformers_"):
            raise ValueError(
                "Le preprocessor n'a pas encore été fitted. Appelez fit() avant get_feature_names()."
            )

        # Récupérer les noms des features numériques
        numeric_features = self.preprocessor.named_transformers_[
            "num"
        ].get_feature_names_out()

        # Récupérer les noms des features catégorielles encodées
        categorical_features = self.preprocessor.named_transformers_[
            "cat"
        ].get_feature_names_out()

        # Combiner les noms des features
        return list(numeric_features) + list(categorical_features)
