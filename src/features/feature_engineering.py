import pandas as pd
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère de nouvelles features à partir du dataset de base.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données de base
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features ajoutées
    """
    logger.info("Début de la génération des features...")
    
    # Créer une copie du DataFrame pour ne pas modifier l'original
    df_features = df.copy()
    
    # 1. Features basées sur les ratios
    df_features['rooms_per_household'] = df_features['AveRooms'] * df_features['AveOccup']
    df_features['bedrooms_per_room'] = df_features['AveBedrms'] / df_features['AveRooms']
    df_features['population_per_household'] = df_features['AveOccup']
    
    # 2. Features basées sur les interactions
    df_features['income_per_room'] = df_features['median_income'] / (df_features['AveRooms'] * df_features['AveOccup'])
    df_features['income_per_person'] = df_features['median_income'] / df_features['population']
    
    # 3. Features basées sur la localisation
    df_features['distance_to_coast'] = np.sqrt(
        (df_features['Longitude'] - (-118.0))**2 + 
        (df_features['Latitude'] - 34.0)**2
    )
    
    # 4. Features basées sur les catégories de prix
    df_features['price_category'] = pd.cut(
        df_features['median_house_value'],
        bins=[0, 150000, 300000, 450000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Luxury']
    )
    
    # 5. Features basées sur l'âge du logement
    df_features['age_category'] = pd.cut(
        df_features['housing_median_age'],
        bins=[0, 20, 40, 60, float('inf')],
        labels=['New', 'Recent', 'Old', 'Very Old']
    )
    
    # 6. Features basées sur la densité
    df_features['population_density'] = df_features['population'] / (df_features['AveRooms'] * df_features['AveOccup'])
    
    logger.info("Génération des features terminée")
    logger.info(f"Nouvelles colonnes ajoutées: {list(set(df_features.columns) - set(df.columns))}")
    
    return df_features 