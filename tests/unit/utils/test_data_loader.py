import pytest
import pandas as pd
from src.data.data_loader import DataLoader
from unittest.mock import patch, mock_open

def test_load_data_success(tmp_path):
    """Test le chargement réussi des données."""
    # Créer un fichier CSV temporaire avec toutes les colonnes attendues
    csv_content = (
        "median_income,housing_median_age,AveRooms,AveBedrms,population,AveOccup,"
        "Latitude,Longitude,median_house_value,ocean_proximity\n"
        "8.3252,41.0,6.984126984126984,1.0238095238095237,322.0,2.5555555555555554,"
        "37.88,-122.23,452600.0,NEAR OCEAN"
    )
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    
    # Instancier le DataLoader
    loader = DataLoader()
    
    # Charger les données
    df = loader.load_data(str(csv_file))
    
    # Vérifications
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 10)
    assert 'Longitude' in df.columns
    assert 'Latitude' in df.columns
    assert 'median_house_value' in df.columns

def test_load_data_file_not_found():
    """Test la gestion d'erreur quand le fichier n'existe pas."""
    loader = DataLoader()
    
    with pytest.raises(FileNotFoundError):
        loader.load_data("fichier_inexistant.csv")

def test_load_data_invalid_format(tmp_path):
    """Test la gestion d'erreur pour un format de fichier invalide."""
    # Créer un fichier texte invalide
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("Ceci n'est pas un CSV valide")
    
    loader = DataLoader()
    
    with pytest.raises(pd.errors.EmptyDataError):
        loader.load_data(str(invalid_file))

@patch('pandas.read_csv')
def test_load_data_with_mock(mock_read_csv, sample_data):
    """Test le chargement des données en utilisant un mock."""
    # Configurer le mock
    mock_read_csv.return_value = sample_data
    
    # Charger les données
    loader = DataLoader()
    df = loader.load_data("dummy.csv")
    
    # Vérifications
    assert isinstance(df, pd.DataFrame)
    assert df.equals(sample_data)
    mock_read_csv.assert_called_once() 