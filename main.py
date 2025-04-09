import logging
import os
import argparse
import sys
from src.pipeline import Pipeline


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
        "--model_type",
        type=str,
        default="random_forest",
        choices=["linear_regression", "random_forest", "xgboost"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help="Whether to tune hyperparameters",
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of folds for cross-validation"
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
        # Créer et exécuter le pipeline
        pipeline = Pipeline(
            input_file=args.input_file,
            model_type=args.model_type,
            tune_hyperparameters=args.tune_hyperparameters,
            cv_folds=args.cv_folds,
        )
        pipeline.run()

        logger.info("Pipeline terminé avec succès!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
