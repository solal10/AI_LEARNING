import logging
import os
from src.utils.data_loader import DataLoader
from src.utils.data_cleaner import DataCleaner
from src.utils.data_preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    """Classe principale du pipeline de traitement des données."""
    
    def __init__(self, input_file, output_dir='data/processed'):
        """
        Initialise le pipeline.
        
        Args:
            input_file (str): Chemin vers le fichier d'entrée
            output_dir (str): Dossier de sortie pour les résultats
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        """
        Exécute le pipeline complet.
        """
        try:
            # Step 1: Data Loading
            logger.info("Step 1: Data Loading")
            data = self.data_loader.load_data(self.input_file)
            logger.info(f"Data loaded with shape: {data.shape}")
            
            # Step 2: Data Cleaning
            logger.info("Step 2: Data Cleaning")
            cleaned_data = self.data_cleaner.clean_data(data)
            logger.info(f"Data cleaned with shape: {cleaned_data.shape}")
            
            # Step 3: Data Preparation
            logger.info("Step 3: Data Preparation")
            X_train, X_test, y_train, y_test, preprocessor = self.data_preprocessor.prepare_data(cleaned_data)
            logger.info(f"Data prepared - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            # Step 4: Model Training
            logger.info("Step 4: Model Training")
            model = self.model_trainer.train_model(X_train, y_train, preprocessor)
            logger.info("Model trained successfully")
            
            # Step 5: Model Evaluation
            logger.info("Step 5: Model Evaluation")
            metrics = self.model_trainer.evaluate_model(X_test, y_test)
            logger.info(f"Model evaluation metrics: {metrics}")
            
            # Step 6: Save Results
            logger.info("Step 6: Saving Results")
            self.model_trainer.save_model('final_model')
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise 