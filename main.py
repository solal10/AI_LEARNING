import logging
import os
import argparse
from datetime import datetime
from src.utils.data_loader import DataLoader
from src.utils.data_cleaner import DataCleaner
from src.utils.data_preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.reporting.report_generator import ReportGenerator

def setup_logging():
    """Configure le logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description='Pipeline d\'analyse du California Housing Dataset')
    
    parser.add_argument('--input_file', type=str, required=True,
                      help='Chemin vers le fichier d\'entrée')
    
    parser.add_argument('--model_type', type=str, default='linear',
                      choices=['linear', 'random_forest', 'xgboost'],
                      help='Type de modèle à utiliser (default: linear)')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion des données pour le test (default: 0.2)')
    
    parser.add_argument('--random_state', type=int, default=42,
                      help='Seed pour la reproductibilité (default: 42)')
    
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Dossier de sortie pour les données traitées (default: data/processed)')
    
    parser.add_argument('--tune_hyperparams', action='store_true',
                      help='Effectuer une recherche d\'hyperparamètres')
    
    parser.add_argument('--n_estimators', type=int, default=100,
                      help='Nombre d\'arbres pour Random Forest ou XGBoost (default: 100)')
    
    parser.add_argument('--max_depth', type=int, default=None,
                      help='Profondeur maximale des arbres (default: None)')
    
    parser.add_argument('--learning_rate', type=float, default=0.1,
                      help='Taux d\'apprentissage pour XGBoost (default: 0.1)')
    
    return parser.parse_args()

def main():
    """Fonction principale du pipeline."""
    # Parser les arguments
    args = parse_args()
    
    # Configurer le logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Créer les dossiers nécessaires
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # === ÉTAPE 1: CHARGEMENT DES DONNÉES ===
        logger.info("=== ÉTAPE 1: CHARGEMENT DES DONNÉES ===")
        data_loader = DataLoader()
        df = data_loader.load_data(args.input_file)
        
        # === ÉTAPE 2: NETTOYAGE DES DONNÉES ===
        logger.info("\n=== ÉTAPE 2: NETTOYAGE DES DONNÉES ===")
        data_cleaner = DataCleaner()
        df_cleaned = data_cleaner.clean_data(df)
        
        # Sauvegarder les données nettoyées
        output_file = os.path.join(args.output_dir, 'california_housing_cleaned.csv')
        df_cleaned.to_csv(output_file, index=False)
        logger.info(f"Données nettoyées sauvegardées dans {output_file}")
        
        # === ÉTAPE 3: PRÉPARATION DES DONNÉES ===
        logger.info("\n=== ÉTAPE 3: PRÉPARATION DES DONNÉES ===")
        data_preprocessor = DataPreprocessor(
            test_size=args.test_size,
            random_state=args.random_state
        )
        X_train, X_test, y_train, y_test, preprocessor = data_preprocessor.prepare_data(df_cleaned)
        
        # Préparer les hyperparamètres
        hyperparams = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'random_state': args.random_state
        }
        
        if args.model_type == 'xgboost':
            hyperparams['learning_rate'] = args.learning_rate
        
        # === ÉTAPE 4: ENTRAÎNEMENT DU MODÈLE ===
        logger.info("\n=== ÉTAPE 4: ENTRAÎNEMENT DU MODÈLE ===")
        model_trainer = ModelTrainer(
            model_type=args.model_type,
            hyperparams=hyperparams
        )
        model = model_trainer.train_model(
            X_train, 
            y_train, 
            preprocessor,
            tune_hyperparams=args.tune_hyperparams
        )
        
        # === ÉTAPE 5: ÉVALUATION DU MODÈLE ===
        logger.info("\n=== ÉTAPE 5: ÉVALUATION DU MODÈLE ===")
        metrics = model_trainer.evaluate_model(X_test, y_test)
        
        # === ÉTAPE 6: GÉNÉRATION DU RAPPORT ===
        logger.info("\n=== ÉTAPE 6: GÉNÉRATION DU RAPPORT ===")
        feature_names = data_preprocessor.get_feature_names(X_train)
        report_generator = ReportGenerator()
        report_generator.generate_report(
            model_trainer,
            X_test,
            y_test,
            feature_names,
            metrics
        )
        
        # === ÉTAPE 7: SAUVEGARDE DU MODÈLE ===
        logger.info("\n=== ÉTAPE 7: SAUVEGARDE DU MODÈLE ===")
        model_trainer.save_model()
        
        logger.info("\nPipeline terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Une erreur est survenue: {str(e)}")
        raise

if __name__ == "__main__":
    main() 