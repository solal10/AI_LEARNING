import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from sklearn.datasets import fetch_california_housing

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Étape 1 — Préparation
print("Étape 1 — Préparation")
# Charger le dataset California Housing
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Créer un DataFrame avec les noms de colonnes
df = pd.DataFrame(X, columns=housing.feature_names)
df["median_house_value"] = y  # Ajouter la variable cible (prix des maisons)

# Renommer les colonnes pour correspondre à nos besoins
df = df.rename(
    columns={
        "MedInc": "median_income",
        "Population": "population",
        "HouseAge": "housing_median_age",
    }
)

# Ajouter une colonne ocean_proximity aléatoire
np.random.seed(42)
ocean_proximity = np.random.choice(
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"], size=len(df)
)
df["ocean_proximity"] = ocean_proximity

# Ajouter une colonne households (nombre de ménages) pour l'estimation de l'aire
np.random.seed(42)
df["households"] = np.random.randint(100, 1000, size=len(df))

print("Aperçu du DataFrame:")
print(df.head())
print("\nInformations sur le DataFrame:")
print(df.info())

# Étape 2 — Feature Engineering
print("\nÉtape 2 — Feature Engineering")

# Créer la colonne price_per_person
df["price_per_person"] = df["median_house_value"] * 100000 / df["population"]

# Créer la colonne area_sqft (estimation)
df["area_sqft"] = df["households"] * 60

# Créer la colonne high_end
df["high_end"] = df["median_house_value"].apply(lambda x: "Oui" if x > 3.5 else "Non")

# Créer la colonne price_category avec pd.cut()
df["price_category"] = pd.cut(
    df["median_house_value"],
    bins=[0, 1.5, 2.5, 3.5, 5.0],
    labels=["Bas", "Moyen", "Élevé", "Luxe"],
)

print("Nouvelles colonnes créées:")
print(
    df[
        [
            "median_house_value",
            "price_per_person",
            "area_sqft",
            "high_end",
            "price_category",
        ]
    ].head()
)

# Étape 3 — GroupBy & Aggregation
print("\nÉtape 3 — GroupBy & Aggregation")

# Moyenne median_income et price_per_person par ocean_proximity et high_end
grouped_stats = (
    df.groupby(["ocean_proximity", "high_end"])
    .agg({"median_income": "mean", "price_per_person": "mean"})
    .round(2)
)
print("Moyenne median_income et price_per_person par ocean_proximity et high_end:")
print(grouped_stats)

# Répartition du price_category par ocean_proximity
price_category_dist = pd.crosstab(df["ocean_proximity"], df["price_category"])
print("\nRépartition du price_category par ocean_proximity:")
print(price_category_dist)

# Étape 4 — Visualisation
print("\nÉtape 4 — Visualisation")

# Configuration du style pour les visualisations
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

# 1. Barplot de median_income par high_end et ocean_proximity
plt.figure(figsize=(12, 6))
pivot_income = df.pivot_table(
    values="median_income", index="ocean_proximity", columns="high_end", aggfunc="mean"
).round(2)
pivot_income.plot(kind="bar")
plt.title("Revenu médian par proximité à l'océan et catégorie haut de gamme")
plt.ylabel("Revenu médian")
plt.xlabel("Proximité à l'océan")
plt.legend(title="Haut de gamme")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("median_income_by_ocean_and_high_end.png")
plt.show()

# 2. Heatmap de corrélations
plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
correlation = df[numeric_cols].corr().round(2)
sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 3. Boxplot du price_per_person par price_category
plt.figure(figsize=(10, 6))
sns.boxplot(x="price_category", y="price_per_person", data=df)
plt.title("Prix par personne par catégorie de prix")
plt.ylabel("Prix par personne ($)")
plt.xlabel("Catégorie de prix")
plt.tight_layout()
plt.savefig("price_per_person_by_category.png")
plt.show()

# Bonus — Pivot table croisant price_category & high_end avec moyenne du revenu médian
print(
    "\nBonus — Pivot table croisant price_category & high_end avec moyenne du revenu médian:"
)
bonus_pivot = pd.pivot_table(
    df,
    values="median_income",
    index="price_category",
    columns="high_end",
    aggfunc="mean",
    fill_value=0,
).round(2)
print(bonus_pivot)

# Bonus — Merge sur une version filtrée du dataset
print("\nBonus — Merge sur une version filtrée du dataset:")

# Créer deux DataFrames filtrés
df_prix = df[df["high_end"] == "Oui"][
    ["ocean_proximity", "median_house_value", "price_per_person", "price_category"]
].copy()
df_info = df[df["ocean_proximity"] == "NEAR BAY"][
    ["ocean_proximity", "median_income", "housing_median_age", "AveRooms", "AveBedrms"]
].copy()

# Ajouter un ID pour le merge
df_prix["id"] = range(1, len(df_prix) + 1)
df_info["id"] = range(1, len(df_info) + 1)

# Merge les DataFrames
merged_df = pd.merge(df_prix, df_info, on="id", how="inner")
print("DataFrame fusionné (premiers 5 lignes):")
print(merged_df.head())

# Sauvegarder le DataFrame traité
df.to_csv("california_housing_processed.csv", index=False)
print("\nDataFrame traité sauvegardé dans 'california_housing_processed.csv'")
