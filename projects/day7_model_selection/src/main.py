import logging
from model_selector import ModelSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to demonstrate model selection techniques."""
    # Initialize model selector
    selector = ModelSelector(output_dir="output")
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test = selector.load_and_prepare_data(
            data_path="data/heart_disease.csv",
            target_col="HeartDisease"
        )
        
        # Initialize models
        rf_model = RandomForestClassifier(random_state=42)
        lr_model = LogisticRegression(random_state=42)
        
        # Run k-fold cross-validation
        print("\nRunning k-fold cross-validation for Random Forest...")
        rf_cv_results = selector.run_kfold_cv(X_train, y_train, rf_model)
        
        print("\nRunning k-fold cross-validation for Logistic Regression...")
        lr_cv_results = selector.run_kfold_cv(X_train, y_train, lr_model)
        
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
        print("\nRunning grid search for Random Forest...")
        best_rf_model, best_rf_params = selector.run_grid_search(
            X_train, y_train,
            rf_model,
            rf_param_grid
        )
        
        # Run grid search for Logistic Regression
        print("\nRunning grid search for Logistic Regression...")
        best_lr_model, best_lr_params = selector.run_grid_search(
            X_train, y_train,
            lr_model,
            lr_param_grid
        )
        
        # Plot learning curves
        print("\nPlotting learning curves...")
        selector.plot_learning_curve(best_rf_model, X_train, y_train, "Random Forest Learning Curve")
        selector.plot_learning_curve(best_lr_model, X_train, y_train, "Logistic Regression Learning Curve")
        
        # Make predictions with best models
        rf_predictions = best_rf_model.predict(X_test)
        lr_predictions = best_lr_model.predict(X_test)
        
        # Get probabilities for ROC curve
        rf_proba = best_rf_model.predict_proba(X_test)[:, 1]
        lr_proba = best_lr_model.predict_proba(X_test)[:, 1]
        
        # Plot confusion matrices
        print("\nPlotting confusion matrices...")
        selector.plot_confusion_matrix(y_test, rf_predictions, "Random Forest Confusion Matrix")
        selector.plot_confusion_matrix(y_test, lr_predictions, "Logistic Regression Confusion Matrix")
        
        # Plot ROC curves
        print("\nPlotting ROC curves...")
        selector.plot_roc_curve(y_test, rf_proba, "Random Forest ROC Curve")
        selector.plot_roc_curve(y_test, lr_proba, "Logistic Regression ROC Curve")
        
        print("\nModel selection completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 