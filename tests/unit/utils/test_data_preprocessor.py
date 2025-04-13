import pytest
import pandas as pd
import numpy as np
from src.data.data_preprocessor import DataPreprocessor
from sklearn.compose import ColumnTransformer


def test_init():
    """Test l'initialisation du DataPreprocessor."""
    preprocessor = DataPreprocessor(test_size=0.3, random_state=42)
    assert preprocessor.test_size == 0.3
    assert preprocessor.random_state == 42
    assert preprocessor.preprocessor is None


def test_prepare_data(sample_data):
    """Test la préparation des données."""
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, transformer = preprocessor.prepare_data(
        sample_data
    )

    # Vérifier les dimensions
    assert len(X_train) + len(X_test) == len(sample_data)
    assert isinstance(transformer, ColumnTransformer)

    # Vérifier que les colonnes sont correctement séparées
    assert "median_house_value" not in X_train.columns
    assert isinstance(y_train, pd.Series)


def test_create_preprocessor(sample_data):
    """Test la création du preprocessor."""
    preprocessor = DataPreprocessor()
    numeric_features = ["longitude", "latitude", "housing_median_age"]
    categorical_features = ["ocean_proximity"]

    transformer = preprocessor._create_preprocessor(
        numeric_features, categorical_features
    )

    assert isinstance(transformer, ColumnTransformer)
    assert (
        len(transformer.transformers) == 2
    )  # Utilisation de transformers au lieu de transformers_
    assert transformer.transformers[0][0] == "num"
    assert transformer.transformers[1][0] == "cat"


def test_get_feature_names(sample_data):
    """Test la récupération des noms de features après transformation."""
    preprocessor = DataPreprocessor()
    X = sample_data.drop("median_house_value", axis=1)

    # Préparer les données pour avoir un preprocessor configuré
    X_train, X_test, y_train, y_test, transformer = preprocessor.prepare_data(
        sample_data
    )

    # Fit le preprocessor
    preprocessor.preprocessor.fit(X_train)

    # Obtenir les noms des features
    feature_names = preprocessor.get_feature_names(X)

    # Vérifications
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    # Vérifier que les features numériques sont présentes
    assert any("longitude" in feature.lower() for feature in feature_names)
    # Vérifier que les features catégorielles encodées sont présentes
    assert any("ocean_proximity" in feature.lower() for feature in feature_names)


def test_prepare_data_with_missing_target(sample_data):
    """Test la gestion d'erreur quand la colonne cible est manquante."""
    preprocessor = DataPreprocessor()
    df_without_target = sample_data.drop("median_house_value", axis=1)

    with pytest.raises(KeyError):
        preprocessor.prepare_data(df_without_target)


def test_prepare_data_empty_dataframe():
    """Test la gestion d'erreur avec un DataFrame vide."""
    preprocessor = DataPreprocessor()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Le DataFrame est vide"):
        preprocessor.prepare_data(empty_df)
