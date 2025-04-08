# Projet de Modélisation - California Housing

Ce projet implémente un pipeline complet de modélisation pour prédire les prix des maisons en Californie en utilisant le dataset California Housing.

## Structure du Projet

```
.
├── data/
│   ├── raw/           # Données brutes
│   └── processed/     # Données nettoyées
├── models/            # Modèles sauvegardés
├── reports/           # Rapports et visualisations
├── src/
│   ├── utils/         # Utilitaires
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   └── data_preprocessor.py
│   ├── models/        # Modèles
│   │   └── model_trainer.py
│   └── reporting/     # Génération de rapports
│       └── report_generator.py
└── main.py           # Script principal
```

## Installation

1. Cloner le repository :
```bash
git clone <repository-url>
cd california-housing
```

2. Créer un environnement virtuel et l'activer :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Placer votre fichier de données dans le dossier `data/raw/`

2. Exécuter le pipeline :
```bash
python main.py --input_file votre_fichier.csv
```

Options disponibles :
- `--input_file` : Nom du fichier d'entrée (défaut: california_housing.csv)
- `--test_size` : Proportion du dataset pour les tests (défaut: 0.2)
- `--random_state` : Seed pour la reproductibilité (défaut: 42)

## Fonctionnalités

1. **Chargement des données**
   - Vérification des colonnes attendues
   - Validation des types de données

2. **Nettoyage des données**
   - Gestion des valeurs manquantes
   - Suppression des doublons
   - Correction des types de données
   - Détection et traitement des outliers

3. **Préparation des données**
   - Séparation features/target
   - Encodage des variables catégorielles
   - Split train/test
   - Standardisation des données

4. **Modélisation**
   - Entraînement d'un modèle de régression
   - Évaluation avec RMSE, MAE, R²
   - Analyse de l'importance des features

5. **Reporting**
   - Génération de visualisations
   - Rapport de performance
   - Sauvegarde des résultats

## Résultats

Les résultats sont sauvegardés dans les dossiers suivants :
- `data/processed/` : Données nettoyées
- `models/` : Modèle entraîné
- `reports/` : Visualisations et rapports

## Dépendances

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

## Auteur

[Votre nom]

## Licence

[Votre licence] 