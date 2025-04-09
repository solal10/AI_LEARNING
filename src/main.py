import logging
import os
import argparse
from src.pipeline import Pipeline

def setup_logging():
    """Configure le logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description='Pipeline de traitement des données California Housing')
    parser.add_argument('--input_file', type=str, default='data/raw/california_housing.csv',
                      help='Chemin vers le fichier de données')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Dossier de sortie pour les résultats')
    return parser.parse_args()

def main():
    """Fonction principale du pipeline."""
    # Parser les arguments
    args = parse_args()
    
    # Configurer le logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Créer et exécuter le pipeline
        pipeline = Pipeline(input_file=args.input_file, output_dir=args.output_dir)
        pipeline.run()
        
        logger.info("Pipeline terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Une erreur est survenue: {str(e)}")
        raise

if __name__ == "__main__":
    main() 