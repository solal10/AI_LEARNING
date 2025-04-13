import pandas as pd
import numpy as np
import logging
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Classe pour nettoyer et préparer les données."""

    def __init__(self):
        """Initialize the data cleaner."""
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values, duplicates, and outliers."""
        try:
            # Create a copy of the dataframe
            df_clean = df.copy()
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_clean)
            
            # Handle duplicates
            df_clean = self._handle_duplicates(df_clean)
            
            # Handle outliers
            df_clean = self._handle_outliers(df_clean)
            
            # Convert categorical columns to appropriate types
            df_clean = self._convert_categorical_types(df_clean)
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        self.logger.info("Handling missing values...")
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                self.logger.info(f"Found {missing_count} missing values in column {col}")
                if df[col].dtype in ['int64', 'float64']:
                    median_value = df[col].median()
                    df.loc[:, col] = df[col].fillna(median_value)
                else:
                    mode_value = df[col].mode()[0]
                    df.loc[:, col] = df[col].fillna(mode_value)
                self.logger.info(f"Filled missing values in column {col}")
            
        return df
        
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate rows in the dataset."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.logger.info(f"Found {duplicates} duplicate rows. Removing them...")
            df = df.drop_duplicates()
            
        return df
        
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical columns using IQR method."""
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                self.logger.info(f"Found {len(outliers)} outliers in {col}")
                # Cap the outliers
                df[col] = df[col].clip(lower_bound, upper_bound)
                
        return df
        
    def _convert_categorical_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to appropriate types."""
        categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 
                          'ExerciseAngina', 'ST_Slope']
        
        for col in categorical_cols:
            df[col] = df[col].astype('category')
            
        return df
