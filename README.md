# Heart Failure Prediction Pipeline

This project implements a machine learning pipeline for predicting heart failure using various classification models.

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── data_cleaner.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── classification_trainer.py
│   └── pipeline.py
├── main.py
├── requirements.txt
└── README.md
```

## Features

- Data loading and exploration
- Data cleaning and preprocessing
- Feature engineering
- Multiple model comparison (Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost)
- Model evaluation with various metrics
- Visualization of results
- Model persistence

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd heart-failure-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline with:
```bash
python main.py --input_file path/to/your/data.csv --output_dir output
```

Arguments:
- `--input_file`: Path to the input CSV file (required)
- `--output_dir`: Directory to save output files (default: 'output')

## Output

The pipeline generates:
- Model comparison results in CSV format
- Classification reports
- Confusion matrix plot
- ROC curve plot
- Saved models

## License

MIT License

# AI Learning Project

This repository contains various AI/ML learning projects and experiments.

## Projects

### Day 7: Model Selection and Cross-Validation

Implementation of model selection techniques using scikit-learn, including:
- K-Fold and Stratified K-Fold Cross-Validation
- Grid Search for Hyperparameter Tuning
- Learning Curve Analysis
- Model Performance Evaluation

#### Usage

Run the model selection pipeline:
```bash
python -m src.cli day7
```

This will:
1. Download the heart disease dataset (if not present)
2. Run cross-validation on Random Forest and Logistic Regression models
3. Perform grid search for hyperparameter tuning
4. Generate visualizations (learning curves, confusion matrices, ROC curves)
5. Save the best performing model to `models/best_day7_model.joblib`

#### Results

The pipeline compares Random Forest and Logistic Regression models:
- Cross-validation scores
- Grid search results
- Model performance metrics (accuracy, F1 score)
- Visualizations in `projects/day7_model_selection/output/`

Best model parameters and performance metrics are logged during execution. 