import os
import pytest
import pandas as pd
from src.data.data_loader import DataLoader
from unittest.mock import patch, mock_open


def test_load_data_success(tmp_path):
    """Test le chargement réussi des données."""
    # Créer un fichier CSV temporaire avec toutes les colonnes attendues
    csv_content = (
        "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,"
        "ExerciseAngina,Oldpeak,ST_Slope,HeartDisease\n"
        "50,M,ATA,140,200,0,Normal,150,N,1.0,Up,0\n"
        "60,F,NAP,120,180,1,ST,140,Y,2.0,Flat,1"
    )
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    # Instancier le DataLoader
    loader = DataLoader(str(csv_file))
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_file_not_found():
    """Test la gestion d'erreur quand le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError):
        loader = DataLoader("nonexistent.csv")
        loader.load_data()


def test_load_data_invalid_format(tmp_path):
    """Test la gestion d'erreur pour un format de fichier invalide."""
    # Créer un fichier texte invalide
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("Ceci n'est pas un CSV valide")

    with pytest.raises(Exception):
        loader = DataLoader(str(invalid_file))
        loader.load_data()


@patch("pandas.read_csv")
def test_load_data_with_mock(mock_read_csv):
    """Test le chargement des données en utilisant un mock."""
    # Créer un DataFrame de test avec les colonnes attendues
    sample_data = pd.DataFrame({
        'Age': [50, 60],
        'Sex': ['M', 'F'],
        'ChestPainType': ['ATA', 'NAP'],
        'RestingBP': [140, 120],
        'Cholesterol': [200, 180],
        'FastingBS': [0, 1],
        'RestingECG': ['Normal', 'ST'],
        'MaxHR': [150, 140],
        'ExerciseAngina': ['N', 'Y'],
        'Oldpeak': [1.0, 2.0],
        'ST_Slope': ['Up', 'Flat'],
        'HeartDisease': [0, 1]
    })
    
    # Configurer le mock
    mock_read_csv.return_value = sample_data

    # Charger les données
    loader = DataLoader("dummy.csv")
    df = loader.load_data()
    assert df.equals(sample_data)
