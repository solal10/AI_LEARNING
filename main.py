import logging
import os
import argparse
import sys
from src.pipeline import Pipeline
from src.features.feature_engineering import generate_features
from src.models.classification_trainer import ClassificationTrainer


def setup_logging():
    """Configure le logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Run the machine learning pipeline")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input data file"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Type de tâche (régression ou classification)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        help="Type de modèle à utiliser",
    )
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help="Activer le tuning des hyperparamètres",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Nombre de folds pour la validation croisée",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output files",
    )
    return parser.parse_args()


def main():
    """
    Main function to run the pipeline.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the heart failure prediction pipeline")
    parser.add_argument("--input_file", type=str, required=True,
                      help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save output files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configurer le logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize and run pipeline
        pipeline = Pipeline(args.input_file, args.output_dir)
        pipeline.run()
        logger.info("Pipeline terminé avec succès!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
