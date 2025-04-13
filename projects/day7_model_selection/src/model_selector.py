import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
    GridSearchCV,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging
import os
from typing import Tuple, Dict, Any, List, Union, Optional
import joblib

class ModelSelector:
    """Class for model selection and validation techniques."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the ModelSelector.
        
        Args:
            output_dir (str): Directory to save outputs and visualizations.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_and_prepare_data(
        self, 
        data_path: str, 
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare data for model selection.
        
        Args:
            data_path (str): Path to the dataset.
            target_col (str): Name of the target column.
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test.
        """
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Preprocess categorical variables
            categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = pd.Categorical(df[col]).codes
            
            # Separate features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                stratify=y,
                random_state=random_state
            )
            
            self.logger.info(f"Data loaded and split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def run_kfold_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        n_splits: int = 5,
        stratified: bool = True
    ) -> Dict[str, List[float]]:
        """Run k-fold cross-validation.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            model: Scikit-learn model instance.
            n_splits (int): Number of folds.
            stratified (bool): Whether to use StratifiedKFold.
            
        Returns:
            Dict containing accuracy and F1 scores for each fold.
        """
        try:
            # Initialize cross-validator
            if stratified:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
            # Store results
            results = {
                'accuracy': [],
                'f1': []
            }
            
            # Run cross-validation
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                
                results['accuracy'].append(accuracy)
                results['f1'].append(f1)
                
                self.logger.info(f"Fold {fold + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
                
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(self.output_dir, 'cv_results.csv'), index=False)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise
            
    def run_grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        param_grid: Dict[str, List[Any]],
        cv: int = 5
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run grid search for hyperparameter tuning.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            model: Scikit-learn model instance.
            param_grid (Dict): Parameter grid for grid search.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Tuple containing best model and best parameters.
        """
        try:
            # Initialize grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
            
            # Run grid search
            grid_search.fit(X, y)
            
            # Log results
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
            
            # Save results
            results_df = pd.DataFrame(grid_search.cv_results_)
            results_df.to_csv(os.path.join(self.output_dir, 'grid_search_results.csv'), index=False)
            
            return grid_search.best_estimator_, grid_search.best_params_
            
        except Exception as e:
            self.logger.error(f"Error in grid search: {str(e)}")
            raise
            
    def plot_learning_curve(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "Learning Curve"
    ) -> None:
        """Plot learning curve for model.
        
        Args:
            model: Scikit-learn model instance.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            title (str): Plot title.
        """
        try:
            # Calculate learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, label='Training score')
            plt.plot(train_sizes, val_mean, label='Cross-validation score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
            
            plt.title(title)
            plt.xlabel('Training examples')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'learning_curve.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting learning curve: {str(e)}")
            raise
            
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix"
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.
            title (str): Plot title.
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(title)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
            
    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve"
    ) -> None:
        """Plot ROC curve.
        
        Args:
            y_true (pd.Series): True labels.
            y_pred_proba (np.ndarray): Predicted probabilities.
            title (str): Plot title.
        """
        try:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
            
    def run_pipeline(
        self,
        data_path: str,
        target_col: str,
        save_best_model_path: Optional[str] = None
    ) -> Any:
        """Run the complete model selection pipeline.
        
        Args:
            data_path (str): Path to the dataset.
            target_col (str): Name of the target column.
            save_best_model_path (str, optional): Path to save the best model.
            
        Returns:
            The best performing model.
        """
        try:
            # Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data(
                data_path=data_path,
                target_col=target_col
            )
            
            # Initialize models
            rf_model = RandomForestClassifier(random_state=42)
            lr_model = LogisticRegression(random_state=42)
            
            # Run k-fold cross-validation
            self.logger.info("\nRunning k-fold cross-validation for Random Forest...")
            rf_cv_results = self.run_kfold_cv(X_train, y_train, rf_model)
            
            self.logger.info("\nRunning k-fold cross-validation for Logistic Regression...")
            lr_cv_results = self.run_kfold_cv(X_train, y_train, lr_model)
            
            # Define parameter grids for grid search
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            lr_param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            # Run grid search for Random Forest
            self.logger.info("\nRunning grid search for Random Forest...")
            best_rf_model, best_rf_params = self.run_grid_search(
                X_train, y_train,
                rf_model,
                rf_param_grid
            )
            
            # Run grid search for Logistic Regression
            self.logger.info("\nRunning grid search for Logistic Regression...")
            best_lr_model, best_lr_params = self.run_grid_search(
                X_train, y_train,
                lr_model,
                lr_param_grid
            )
            
            # Plot learning curves
            self.logger.info("\nPlotting learning curves...")
            self.plot_learning_curve(best_rf_model, X_train, y_train, "Random Forest Learning Curve")
            self.plot_learning_curve(best_lr_model, X_train, y_train, "Logistic Regression Learning Curve")
            
            # Make predictions with best models
            rf_predictions = best_rf_model.predict(X_test)
            lr_predictions = best_lr_model.predict(X_test)
            
            # Get probabilities for ROC curve
            rf_proba = best_rf_model.predict_proba(X_test)[:, 1]
            lr_proba = best_lr_model.predict_proba(X_test)[:, 1]
            
            # Plot confusion matrices
            self.logger.info("\nPlotting confusion matrices...")
            self.plot_confusion_matrix(y_test, rf_predictions, "Random Forest Confusion Matrix")
            self.plot_confusion_matrix(y_test, lr_predictions, "Logistic Regression Confusion Matrix")
            
            # Plot ROC curves
            self.logger.info("\nPlotting ROC curves...")
            self.plot_roc_curve(y_test, rf_proba, "Random Forest ROC Curve")
            self.plot_roc_curve(y_test, lr_proba, "Logistic Regression ROC Curve")
            
            # Compare models and select the best one
            rf_score = f1_score(y_test, rf_predictions)
            lr_score = f1_score(y_test, lr_predictions)
            
            best_model = best_rf_model if rf_score > lr_score else best_lr_model
            model_name = "Random Forest" if rf_score > lr_score else "Logistic Regression"
            
            self.logger.info(f"\nBest model: {model_name} (F1 Score: {max(rf_score, lr_score):.4f})")
            
            # Save best model if path provided
            if save_best_model_path:
                os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)
                joblib.dump(best_model, save_best_model_path)
                self.logger.info(f"Best model saved to: {save_best_model_path}")
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            raise 