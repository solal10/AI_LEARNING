import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import os
import joblib
from datetime import datetime

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.features.feature_engineering import FeatureEngineer
from src.models.classification_trainer import ClassificationTrainer

class Pipeline:
    def __init__(self, input_file: str, output_dir: str = "output"):
        """Initialize the pipeline with input file and output directory."""
        self.input_file = input_file
        self.output_dir = output_dir
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories for output."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
    def run(self):
        """Run the complete pipeline."""
        try:
            self.logger.info("Starting pipeline...")
            
            # Step 1: Load data
            self.logger.info("Step 1: Loading data...")
            data_loader = DataLoader(self.input_file)
            df = data_loader.load_data()
            
            # Step 2: Clean data
            self.logger.info("Step 2: Cleaning data...")
            data_cleaner = DataCleaner()
            df_clean = data_cleaner.clean_data(df)
            
            # Step 3: Feature engineering
            self.logger.info("Step 3: Engineering features...")
            feature_engineer = FeatureEngineer()
            df_engineered = feature_engineer.engineer_features(df_clean)
            
            # Step 4: Prepare data
            self.logger.info("Step 4: Preparing data...")
            X_train, X_test, y_train, y_test = self.prepare_data(df_engineered)
            
            # Step 5: Train and evaluate models
            self.logger.info("Step 5: Training and evaluating models...")
            trainer = ClassificationTrainer()
            results = trainer.compare_models(X_train, X_test, y_train, y_test)
            
            # Step 6: Generate reports and plots
            self.logger.info("Step 6: Generating reports and plots...")
            self.generate_reports(results, y_test)
            
            # Save best model
            best_model = results['best_model']['model']
            model_path = os.path.join(self.output_dir, "models", "best_model.joblib")
            joblib.dump(best_model, model_path)
            self.logger.info(f"Best model saved to {model_path}")
            
            self.logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
            
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training by splitting into train and test sets."""
        # Separate features and target
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"Training set shape: {X_train.shape}")
        self.logger.info(f"Testing set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    def generate_reports(self, results: Dict[str, Any], y_test: pd.Series):
        """Generate reports and plots for model evaluation."""
        # Save results to CSV
        metrics_df = pd.DataFrame({
            'Model': list(results.keys())[:-2],  # Exclude 'best_model' and 'best_model_name'
            'Accuracy': [results[model]['accuracy'] for model in results if model not in ['best_model', 'best_model_name']],
            'Precision': [results[model]['precision'] for model in results if model not in ['best_model', 'best_model_name']],
            'Recall': [results[model]['recall'] for model in results if model not in ['best_model', 'best_model_name']],
            'F1-score': [results[model]['f1'] for model in results if model not in ['best_model', 'best_model_name']],
            'AUC': [results[model]['auc'] for model in results if model not in ['best_model', 'best_model_name']]
        })
        metrics_df = metrics_df.sort_values('F1-score', ascending=False)
        metrics_df.to_csv(os.path.join(self.output_dir, "reports", "model_comparison.csv"), index=False)
        
        # Generate classification report
        best_predictions = results['best_model']['predictions']
        report = classification_report(y_test, best_predictions)
        with open(os.path.join(self.output_dir, "reports", "classification_report.txt"), "w") as f:
            f.write(f"Best Model: {results['best_model_name']}\n\n")
            f.write("Classification Report\n")
            f.write("====================\n\n")
            f.write(report)
        
        # Generate confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {results["best_model_name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.output_dir, "plots", "confusion_matrix.png"))
        plt.close()
        
        # Generate ROC curves
        plt.figure(figsize=(10, 8))
        for model_name, metrics in results.items():
            if model_name not in ['best_model', 'best_model_name']:
                fpr, tpr, _ = metrics['roc_curve']
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "plots", "roc_curves.png"))
        plt.close()
