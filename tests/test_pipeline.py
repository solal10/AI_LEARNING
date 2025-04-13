import os
import pytest
import pandas as pd
import numpy as np
from src.pipeline import Pipeline
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.features.feature_engineering import FeatureEngineer
from src.models.classification_trainer import ClassificationTrainer

def test_data_loading(tmp_path):
    """Test data loading functionality."""
    # Create a sample dataset
    data = {
        'Age': [50, 60, 70],
        'Sex': ['M', 'F', 'M'],
        'ChestPainType': ['ATA', 'NAP', 'ATA'],
        'RestingBP': [140, 120, 130],
        'Cholesterol': [200, 180, 220],
        'FastingBS': [0, 1, 0],
        'RestingECG': ['Normal', 'ST', 'Normal'],
        'MaxHR': [150, 140, 130],
        'ExerciseAngina': ['N', 'Y', 'N'],
        'Oldpeak': [1.0, 2.0, 1.5],
        'ST_Slope': ['Up', 'Flat', 'Up'],
        'HeartDisease': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    test_file = tmp_path / "test_data.csv"
    df.to_csv(test_file, index=False)
    
    # Test data loader
    loader = DataLoader(str(test_file))
    df_loaded = loader.load_data()
    assert isinstance(df_loaded, pd.DataFrame)
    assert not df_loaded.empty
    assert df_loaded.shape == df.shape

def test_data_cleaning():
    """Test data cleaning functionality."""
    # Create a sample dataset with missing values and outliers
    data = {
        'Age': [50, 60, np.nan],
        'Sex': ['M', 'F', 'M'],
        'ChestPainType': ['ATA', 'NAP', 'ATA'],
        'RestingBP': [140, 1000, 130],  # Outlier
        'Cholesterol': [200, 180, np.nan],
        'FastingBS': [0, 1, 0],
        'RestingECG': ['Normal', 'ST', 'Normal'],
        'MaxHR': [150, 140, 130],
        'ExerciseAngina': ['N', 'Y', 'N'],
        'Oldpeak': [1.0, 2.0, 1.5],
        'ST_Slope': ['Up', 'Flat', 'Up'],
        'HeartDisease': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Test data cleaner
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df)
    assert isinstance(df_clean, pd.DataFrame)
    assert not df_clean.empty
    assert not df_clean.isnull().any().any()

def test_feature_engineering():
    """Test feature engineering functionality."""
    # Create a sample dataset
    data = {
        'Age': [50, 60, 70],
        'Sex': ['M', 'F', 'M'],
        'ChestPainType': ['ATA', 'NAP', 'ATA'],
        'RestingBP': [140, 120, 130],
        'Cholesterol': [200, 180, 220],
        'FastingBS': [0, 1, 0],
        'RestingECG': ['Normal', 'ST', 'Normal'],
        'MaxHR': [150, 140, 130],
        'ExerciseAngina': ['N', 'Y', 'N'],
        'Oldpeak': [1.0, 2.0, 1.5],
        'ST_Slope': ['Up', 'Flat', 'Up'],
        'HeartDisease': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Test feature engineer
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    assert isinstance(df_engineered, pd.DataFrame)
    assert not df_engineered.empty
    assert len(df_engineered.columns) > len(df.columns)

def test_model_training(tmp_path):
    """Test model training functionality."""
    # Create a sample dataset with balanced classes
    data = {
        'Age': [50, 60, 70, 55, 65, 45, 75, 52, 63, 58, 48, 62, 72, 53, 67, 47, 73, 54, 64, 59],
        'Sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'ChestPainType': ['ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP'],
        'RestingBP': [140, 120, 130, 125, 135, 115, 145, 122, 128, 132, 138, 118, 133, 127, 137, 117, 142, 124, 129, 134],
        'Cholesterol': [200, 180, 220, 190, 210, 170, 230, 185, 205, 195, 215, 175, 225, 188, 208, 172, 228, 182, 202, 198],
        'FastingBS': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        'RestingECG': ['Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST'],
        'MaxHR': [150, 140, 130, 145, 135, 155, 125, 142, 138, 148, 152, 143, 132, 147, 137, 153, 127, 144, 139, 146],
        'ExerciseAngina': ['N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y'],
        'Oldpeak': [1.0, 2.0, 1.5, 1.2, 1.8, 0.8, 2.2, 1.1, 1.7, 1.3, 0.9, 1.9, 1.4, 1.6, 2.1, 0.7, 2.3, 1.0, 1.8, 1.5],
        'ST_Slope': ['Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat'],
        'HeartDisease': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Save to temporary file
    test_file = tmp_path / "test_data.csv"
    df.to_csv(test_file, index=False)

    # Test pipeline
    pipeline = Pipeline(str(test_file), str(tmp_path / "output"))
    pipeline.run()

def test_pipeline(tmp_path):
    """Test the complete pipeline."""
    # Create a sample dataset
    data = {
        'Age': [50, 60, 70, 55, 65, 45, 75, 80, 85, 90],
        'Sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'ChestPainType': ['ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP', 'ATA', 'NAP'],
        'RestingBP': [140, 120, 130, 125, 135, 115, 145, 150, 155, 160],
        'Cholesterol': [200, 180, 220, 190, 210, 170, 230, 240, 250, 260],
        'FastingBS': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        'RestingECG': ['Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST', 'Normal', 'ST'],
        'MaxHR': [150, 140, 130, 145, 135, 155, 125, 120, 115, 110],
        'ExerciseAngina': ['N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y'],
        'Oldpeak': [1.0, 2.0, 1.5, 1.2, 1.8, 0.8, 2.2, 2.5, 2.8, 3.0],
        'ST_Slope': ['Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat', 'Up', 'Flat'],
        'HeartDisease': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    test_file = tmp_path / "test_data.csv"
    df.to_csv(test_file, index=False)
    
    # Test pipeline
    pipeline = Pipeline(str(test_file), str(tmp_path / "output"))
    pipeline.run() 