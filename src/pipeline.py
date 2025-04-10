import logging
import os
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.data_preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.models.classification_trainer import ClassificationTrainer
from src.features.feature_engineering import generate_features

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline class that orchestrates the data processing and model training workflow.
    """

    def __init__(
        self,
        input_file: str,
        model_type: str = "random_forest",
        tune_hyperparameters: bool = False,
        cv_folds: int = 5,
        task: str = "regression",
    ):
        """
        Initialize the pipeline.

        Args:
            input_file (str): Path to the input data file
            model_type (str): Type of model to train
            tune_hyperparameters (bool): Whether to tune hyperparameters
            cv_folds (int): Number of folds for cross-validation
            task (str): Type of task (regression or classification)
        """
        self.input_file = input_file
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.task = task

        # Initialize components
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_preprocessor = DataPreprocessor(task=task)

        # Initialize appropriate trainer based on task
        if task == "regression":
            self.model_trainer = ModelTrainer(
                model_type=model_type,
                hyperparams={} if not tune_hyperparameters else None,
                cv_folds=cv_folds,
                task=task
            )
        else:  # classification
            self.model_trainer = ClassificationTrainer(
                model_type=model_type,
                hyperparams={} if not tune_hyperparameters else None,
                cv_folds=cv_folds,
                task=task
            )

        # Create output directories
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

    def run(self):
        """
        Execute the complete pipeline.
        """
        try:
            logger.info("Starting pipeline...")
            
            # Step 1: Loading data
            logger.info("Step 1: Loading data...")
            data = self.data_loader.load_data(self.input_file)
            logger.info(f"Data loaded with shape: {data.shape}")
            
            # Step 2: Cleaning data
            logger.info("Step 2: Cleaning data...")
            cleaned_data = self.data_cleaner.clean_data(data)
            logger.info(f"Data cleaned with shape: {cleaned_data.shape}")
            
            # Step 2.5: Feature Engineering
            logger.info("Step 2.5: Feature Engineering...")
            engineered_data = generate_features(cleaned_data)
            logger.info(f"Features generated. New shape: {engineered_data.shape}")
            
            # Step 3: Preparing data
            logger.info("Step 3: Preparing data...")
            X_train, X_test, y_train, y_test, preprocessor = self.data_preprocessor.prepare_data(engineered_data)
            logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

            # Check class balance for classification
            if self.task == "classification":
                logger.info("Class distribution:")
                logger.info(y_train.value_counts(normalize=True))
            
            # Step 3.5: Comparing models
            logger.info("Step 3.5: Comparing models...")
            comparison_results = self.model_trainer.compare_models(X_train, y_train, X_test, y_test, preprocessor)
            logger.info("Model comparison completed")
            
            # Step 4: Training model
            logger.info("Step 4: Training model...")
            model = self.model_trainer.train_model(X_train, y_train, preprocessor, self.tune_hyperparameters)
            logger.info("Model training completed")
            
            # Step 4.5: Cross-validation
            logger.info("Step 4.5: Performing cross-validation...")
            cv_metrics = self.model_trainer.cross_validate_model(X_train, y_train)
            logger.info("Cross-validation completed")
            
            # Step 5: Evaluating model
            logger.info("Step 5: Evaluating model...")
            test_metrics = self.model_trainer.evaluate_model(X_test, y_test)
            logger.info("Model evaluation completed")
            
            # Step 6: Saving results
            logger.info("Step 6: Saving results...")
            self.model_trainer.save_model("final_model")
            self.model_trainer.save_results(model, test_metrics, cv_metrics)
            logger.info("Results saved")
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
