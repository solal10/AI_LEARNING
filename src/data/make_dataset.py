import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import os

# Créer le dossier data/raw s'il n'existe pas
os.makedirs("data/raw", exist_ok=True)

# Télécharger le dataset
print("Téléchargement du dataset California Housing...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["median_house_value"] = housing.target

# Renommer les colonnes pour plus de clarté
df = df.rename(
    columns={
        "MedInc": "median_income",
        "Population": "population",
        "HouseAge": "housing_median_age",
    }
)

# Ajouter une colonne ocean_proximity aléatoire pour l'exemple
np.random.seed(42)
ocean_proximity = np.random.choice(
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"], size=len(df)
)
df["ocean_proximity"] = ocean_proximity

# Sauvegarder le dataset
output_file = "data/raw/california_housing.csv"
df.to_csv(output_file, index=False)
print(f"Dataset sauvegardé dans {output_file}")
print(f"Shape du dataset: {df.shape}")
