import pandas as pd
import numpy as np
import logging
from typing import List

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer."""
        self.logger = logging.getLogger(__name__)
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features and transform existing ones."""
        try:
            # Create a copy of the dataframe
            df_engineered = df.copy()
            
            # Create new features
            df_engineered = self._create_new_features(df_engineered)
            
            # Transform categorical features
            df_engineered = self._transform_categorical_features(df_engineered)
            
            # Handle any remaining NaN values
            df_engineered = self._handle_nan_values(df_engineered)
            
            self.logger.info(f"Final dataset shape after feature engineering: {df_engineered.shape}")
            
            return df_engineered
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise
            
    def _create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        # Create age groups
        df['AgeGroup'] = pd.cut(df['Age'], 
                              bins=[0, 30, 40, 50, 60, np.inf],
                              labels=['<30', '30-40', '40-50', '50-60', '60+'])
        
        # Create blood pressure categories
        df['BPCategory'] = pd.cut(df['RestingBP'],
                                bins=[0, 120, 140, 160, np.inf],
                                labels=['Normal', 'Prehypertension', 'Stage1', 'Stage2'])
        
        # Create heart rate categories
        df['HRCategory'] = pd.cut(df['MaxHR'],
                                bins=[0, 100, 120, 140, 160, np.inf],
                                labels=['Very Low', 'Low', 'Normal', 'High', 'Very High'])
        
        # Create cholesterol categories
        df['CholesterolCategory'] = pd.cut(df['Cholesterol'],
                                         bins=[0, 200, 240, np.inf],
                                         labels=['Normal', 'Borderline', 'High'])
        
        # Create interaction features
        df['BP_Age_Interaction'] = df['RestingBP'] * df['Age'] / 100
        df['HR_Age_Interaction'] = df['MaxHR'] * df['Age'] / 100
        
        return df
        
    def _transform_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using appropriate encoding."""
        # Convert binary features to numeric
        binary_mapping = {
            'Sex': {'M': 1, 'F': 0},
            'FastingBS': {1: 1, 0: 0},  # Changed from string to int
            'ExerciseAngina': {'Y': 1, 'N': 0}
        }
        
        for col, mapping in binary_mapping.items():
            df[col] = df[col].map(mapping)
        
        # One-hot encode non-binary categorical features
        categorical_cols = ['ChestPainType', 'RestingECG', 'ST_Slope', 
                          'AgeGroup', 'BPCategory', 'HRCategory', 'CholesterolCategory']
        
        # Create an empty DataFrame for the encoded features
        encoded_features = pd.DataFrame(index=df.index)
        
        for col in categorical_cols:
            if col in df.columns:  # Check if column exists
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col)
                # Add dummy variables to the encoded features
                encoded_features = pd.concat([encoded_features, dummies], axis=1)
        
        # Drop the original categorical columns
        df = df.drop(columns=[col for col in categorical_cols if col in df.columns])
        
        # Combine numerical and encoded features
        df = pd.concat([df, encoded_features], axis=1)
        
        return df
        
    def _handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle any remaining NaN values in the dataset."""
        # For numerical columns, use median imputation
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                self.logger.info(f"Filled NaN values in {col} with median: {median_value}")
        
        # For categorical columns (dummy variables), fill with 0
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
                self.logger.info(f"Filled NaN values in {col} with 0")
        
        return df

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