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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Classe pour prétraiter les données du California Housing Dataset.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialise le DataPreprocessor.
        
        Args:
            test_size (float): Proportion des données pour le test
            random_state (int): Graine aléatoire pour la reproductibilité
        """
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
    
    def prepare_data(self, df):
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df (pd.DataFrame): DataFrame à prétraiter
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessor)
        """
        if df.empty:
            raise ValueError("Le DataFrame est vide")
        
        # Séparer features et target
        target_column = 'median_house_value'
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Identifier les features numériques et catégorielles
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        logger.info(f"Features numériques: {len(numeric_features)}")
        logger.info(f"Features catégorielles: {len(categorical_features)}")
        
        # Créer le preprocessor
        self.preprocessor = self._create_preprocessor(numeric_features, categorical_features)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Taille du dataset d'entraînement: {X_train.shape}")
        logger.info(f"Taille du dataset de test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, self.preprocessor
    
    def _create_preprocessor(self, numeric_features, categorical_features):
        """
        Crée un preprocessor pour transformer les données.
        
        Args:
            numeric_features (list): Liste des features numériques
            categorical_features (list): Liste des features catégorielles
            
        Returns:
            ColumnTransformer: Preprocessor configuré
        """
        numeric_transformer = StandardScaler()
        try:
            # Try new scikit-learn version (>=1.2)
            categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            # Fallback for older scikit-learn versions
            categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
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
        if not hasattr(self.preprocessor, 'named_transformers_'):
            raise ValueError("Le preprocessor n'a pas encore été fitted. Appelez fit() avant get_feature_names().")
        
        # Récupérer les noms des features numériques
        numeric_features = self.preprocessor.named_transformers_['num'].get_feature_names_out()
        
        # Récupérer les noms des features catégorielles encodées
        categorical_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
        
        # Combiner les noms des features
        return list(numeric_features) + list(categorical_features) 