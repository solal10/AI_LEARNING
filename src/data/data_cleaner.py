import pandas as pd
import numpy as np
import logging
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Classe pour nettoyer et préparer les données."""
    
    def __init__(self):
        """Initialise le DataCleaner."""
        pass
    
    def clean_data(self, df):
        """
        Nettoie le DataFrame en appliquant toutes les étapes de nettoyage.
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            
        Returns:
            pd.DataFrame: DataFrame nettoyé
        """
        logger.info("Début du nettoyage des données")
        
        # Copier le DataFrame pour éviter les modifications sur l'original
        df_clean = df.copy()
        
        # 1. Gérer les valeurs manquantes
        df_clean = self._handle_missing_values(df_clean)
        
        # 2. Supprimer les doublons
        df_clean = self._remove_duplicates(df_clean)
        
        # 3. Corriger les types de données
        df_clean = self._fix_dtypes(df_clean)
        
        # 4. Détecter et traiter les outliers
        df_clean = self._handle_outliers(df_clean)
        
        logger.info(f"Nettoyage terminé. Shape final: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df):
        """
        Gère les valeurs manquantes.
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            
        Returns:
            pd.DataFrame: DataFrame avec les valeurs manquantes traitées
        """
        logger.info("Traitement des valeurs manquantes")
        
        # Afficher le nombre de valeurs manquantes par colonne
        missing_counts = df.isnull().sum()
        logger.info(f"Valeurs manquantes par colonne:\n{missing_counts[missing_counts > 0]}")
        
        # Remplir les valeurs manquantes numériques avec la moyenne
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())
        
        # Remplir les valeurs manquantes catégorielles avec le mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _remove_duplicates(self, df):
        """
        Supprime les doublons.
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            
        Returns:
            pd.DataFrame: DataFrame sans doublons
        """
        logger.info("Suppression des doublons")
        n_duplicates = df.duplicated().sum()
        logger.info(f"Nombre de doublons trouvés: {n_duplicates}")
        
        if n_duplicates > 0:
            df = df.drop_duplicates()
            logger.info(f"Doublons supprimés. Nouveau shape: {df.shape}")
        
        return df
    
    def _fix_dtypes(self, df):
        """
        Corrige les types de données.
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            
        Returns:
            pd.DataFrame: DataFrame avec les types corrigés
        """
        logger.info("Correction des types de données")
        
        # Convertir les colonnes appropriées en entiers
        integer_columns = ['housing_median_age', 'population']
        for col in integer_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        return df
    
    def _handle_outliers(self, df):
        """
        Détecte et traite les outliers avec la méthode IQR.
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            
        Returns:
            pd.DataFrame: DataFrame avec les outliers traités
        """
        logger.info("Détection et traitement des outliers")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Calculer les limites IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Compter les outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            n_outliers = len(outliers)
            
            if n_outliers > 0:
                logger.info(f"Colonne {col}: {n_outliers} outliers détectés")
                
                # Remplacer les outliers par les limites
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                
                logger.info(f"Outliers traités pour la colonne {col}")
        
        return df 