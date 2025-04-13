import pandas as pd
import logging
from typing import Union, Tuple
import os
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Classe pour charger les données du California Housing Dataset.
    """

    def __init__(self, input_file: str):
        """Initialize the data loader with input file path."""
        self.input_file = input_file
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load and explore the dataset."""
        try:
            # Load the data
            df = pd.read_csv(self.input_file)
            
            # Expected columns for heart failure dataset
            expected_columns = {
                'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                'Oldpeak', 'ST_Slope', 'HeartDisease'
            }
            
            # Check if all required columns are present
            missing_columns = expected_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in dataset: {missing_columns}")
            
            # Log basic information
            self.logger.info(f"Dataset shape: {df.shape}")
            self.logger.info("\nDataset info:")
            self.logger.info(df.info())
            self.logger.info("\nDataset description:")
            self.logger.info(df.describe())
            self.logger.info("\nFirst few rows:")
            self.logger.info(df.head())
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                self.logger.warning("Missing values found:")
                self.logger.warning(missing_values[missing_values > 0])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _verify_columns(self, df: pd.DataFrame):
        """
        Vérifie que toutes les colonnes attendues sont présentes.

        Args:
            df (pd.DataFrame): DataFrame à vérifier
        """
        missing_columns = set(self.expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Colonnes manquantes: {missing_columns}")
            raise ValueError(f"Colonnes manquantes dans le dataset: {missing_columns}")

    def save_data(self, df: pd.DataFrame, file_path: str):
        """
        Sauvegarde les données dans un fichier CSV.

        Args:
            df (pd.DataFrame): DataFrame à sauvegarder
            file_path (str): Chemin où sauvegarder le fichier
        """
        try:
            # Créer le dossier parent si nécessaire
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # Sauvegarder les données
            df.to_csv(file_path, index=False)
            logger.info(f"Données sauvegardées dans {file_path}")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")
            raise

    def _verify_dtypes(self, df):
        """
        Vérifie les types de données des colonnes.

        Args:
            df (pd.DataFrame): DataFrame à vérifier
        """
        # Vérifier les types numériques
        numeric_columns = [
            "median_income",
            "housing_median_age",
            "AveRooms",
            "AveBedrms",
            "population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "median_house_value",
        ]

        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"La colonne {col} n'est pas numérique")

        # Vérifier le type catégoriel
        if not pd.api.types.is_string_dtype(df["ocean_proximity"]):
            logger.warning("La colonne ocean_proximity n'est pas de type string")
