import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Charger le dataset
print("Chargement du dataset California Housing...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['median_house_value'] = housing.target

# Renommer les colonnes pour plus de clarté
df = df.rename(columns={
    'MedInc': 'median_income',
    'Population': 'population',
    'HouseAge': 'housing_median_age'
})

# Ajouter une colonne ocean_proximity aléatoire pour l'exemple
np.random.seed(42)
ocean_proximity = np.random.choice(
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"], 
    size=len(df)
)
df['ocean_proximity'] = ocean_proximity

print("\n=== ÉTAPE 1: DÉTECTION DES VALEURS MANQUANTES ===")
# Créer artificiellement quelques valeurs manquantes pour l'exemple
np.random.seed(42)
mask = np.random.random(df.shape) < 0.02  # 2% de valeurs manquantes
df_missing = df.copy()
df_missing[mask] = np.nan

print("\nNombre de valeurs manquantes par colonne:")
print(df_missing.isnull().sum())

# Remplir les valeurs manquantes avec la moyenne pour les colonnes numériques
numeric_columns = df_missing.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df_missing[col] = df_missing[col].fillna(df_missing[col].mean())

# Remplir les valeurs manquantes avec le mode pour les colonnes catégorielles
categorical_columns = df_missing.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df_missing[col] = df_missing[col].fillna(df_missing[col].mode()[0])

print("\n=== ÉTAPE 2: DÉTECTION DES DOUBLONS ===")
# Créer artificiellement quelques doublons pour l'exemple
df_duplicates = df_missing.copy()
df_duplicates = pd.concat([df_duplicates, df_duplicates.sample(n=100, random_state=42)])

print("Nombre de doublons avant nettoyage:", df_duplicates.duplicated().sum())
df_cleaned = df_duplicates.drop_duplicates()
print("Nombre de doublons après nettoyage:", df_cleaned.duplicated().sum())

print("\n=== ÉTAPE 3: CORRECTION DES TYPES ===")
print("\nTypes de données avant correction:")
print(df_cleaned.dtypes)

# Convertir les colonnes appropriées en types entiers
df_cleaned['housing_median_age'] = df_cleaned['housing_median_age'].astype(int)
df_cleaned['population'] = df_cleaned['population'].astype(int)

print("\nTypes de données après correction:")
print(df_cleaned.dtypes)

print("\n=== ÉTAPE 4: DÉTECTION DES OUTLIERS ===")
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers), lower_bound, upper_bound

# Détecter les outliers pour plusieurs colonnes numériques
numeric_columns = ['median_income', 'population', 'median_house_value']
for col in numeric_columns:
    n_outliers, lower, upper = detect_outliers(df_cleaned, col)
    print(f"\nOutliers pour {col}:")
    print(f"Nombre d'outliers: {n_outliers}")
    print(f"Limites: [{lower:.2f}, {upper:.2f}]")

# Visualisation des outliers pour median_house_value
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned['median_house_value'])
plt.title('Distribution des prix médians des maisons')
plt.savefig('house_value_outliers.png')
plt.close()

print("\n=== ÉTAPE 5: ENCODAGE DES COLONNES CATÉGORIELLES ===")
# Encoder la colonne ocean_proximity
df_encoded = pd.get_dummies(df_cleaned, columns=['ocean_proximity'], drop_first=True)

print("\nColonnes après encodage:")
print(df_encoded.columns.tolist())

print("\n=== RÉSUMÉ FINAL ===")
print("\nStatistiques descriptives avant nettoyage:")
print(df.describe())

print("\nStatistiques descriptives après nettoyage:")
print(df_encoded.describe())

# Sauvegarder le dataset nettoyé
df_encoded.to_csv('california_housing_cleaned.csv', index=False)
print("\nDataset nettoyé sauvegardé dans 'california_housing_cleaned.csv'") 