import pandas as pd
import logging
from typing import Union, Tuple
import os
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Classe pour charger les données du California Housing Dataset.
    """
    
    def __init__(self):
        """
        Initialise le DataLoader avec les colonnes attendues.
        """
        self.expected_columns = [
            'median_income',
            'housing_median_age',
            'AveRooms',
            'AveBedrms',
            'population',
            'AveOccup',
            'Latitude',
            'Longitude',
            'median_house_value',
            'ocean_proximity'
        ]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.
        
        Args:
            file_path (str): Chemin vers le fichier de données
            
        Returns:
            pd.DataFrame: DataFrame contenant les données
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            pd.errors.EmptyDataError: Si le fichier est vide ou dans un format invalide
            ValueError: Si les colonnes attendues sont manquantes
        """
        try:
            logger.info(f"Chargement du fichier: {file_path}")
            df = pd.read_csv(file_path)
            
            # Vérifier si le DataFrame est vide ou invalide
            if df.empty or len(df.columns) == 0:
                raise pd.errors.EmptyDataError(f"Fichier vide ou format invalide: {file_path}")
            
            self._verify_columns(df)
            return df
            
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {file_path}")
            raise
            
        except pd.errors.EmptyDataError:
            logger.error(f"Fichier vide ou format invalide: {file_path}")
            raise
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
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
        numeric_columns = ['median_income', 'housing_median_age', 'AveRooms', 
                         'AveBedrms', 'population', 'AveOccup', 'Latitude', 
                         'Longitude', 'median_house_value']
        
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"La colonne {col} n'est pas numérique")
        
        # Vérifier le type catégoriel
        if not pd.api.types.is_string_dtype(df['ocean_proximity']):
            logger.warning("La colonne ocean_proximity n'est pas de type string") 