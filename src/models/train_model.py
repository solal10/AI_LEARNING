import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset nettoyé
print("Chargement du dataset nettoyé...")
df = pd.read_csv("california_housing_cleaned.csv")

# Séparer les features (X) et la target (y)
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== SPLIT DES DONNÉES ===")
print(f"Taille du dataset d'entraînement: {X_train.shape}")
print(f"Taille du dataset de test: {X_test.shape}")

# Créer et entraîner le pipeline
print("\n=== CRÉATION ET ENTRAÎNEMENT DU PIPELINE ===")
pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Faire les prédictions
y_pred = pipeline.predict(X_test)

# Calculer les métriques
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== PERFORMANCE DU MODÈLE ===")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Valeurs réelles vs Prédictions")
plt.savefig("predictions_vs_real.png")
plt.close()

# Afficher les coefficients du modèle
feature_names = X.columns
coefficients = pipeline.named_steps["regressor"].coef_
feature_importance = pd.DataFrame(
    {"Feature": feature_names, "Coefficient": coefficients}
)
feature_importance = feature_importance.sort_values(
    "Coefficient", key=abs, ascending=False
)

print("\n=== IMPORTANCE DES FEATURES ===")
print(feature_importance)

# Visualisation de l'importance des features
plt.figure(figsize=(12, 6))
sns.barplot(x="Coefficient", y="Feature", data=feature_importance)
plt.title("Importance des features (coefficients de régression)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Afficher quelques prédictions détaillées
print("\n=== EXEMPLES DE PRÉDICTIONS ===")
predictions_df = pd.DataFrame(
    {"Valeur réelle": y_test, "Prédiction": y_pred, "Erreur": np.abs(y_test - y_pred)}
)
print(predictions_df.head(10))
