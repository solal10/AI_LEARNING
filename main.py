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
    return parser.parse_args()


def main():
    """
    Main function to run the pipeline.
    """
    # Parser les arguments
    args = parse_args()

    # Configurer le logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        if args.task == "regression":
            pipeline = Pipeline(
                input_file=args.input_file,
                model_type=args.model_type,
                tune_hyperparameters=args.tune_hyperparameters,
                cv_folds=args.cv_folds,
            )
        else:  # classification
            pipeline = Pipeline(
                input_file=args.input_file,
                model_type=args.model_type,
                tune_hyperparameters=args.tune_hyperparameters,
                cv_folds=args.cv_folds,
                task="classification",
            )

        pipeline.run()
        logger.info("Pipeline terminé avec succès!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
