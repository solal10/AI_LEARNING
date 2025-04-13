# Day 7 - Model Selection and Validation

This project demonstrates various model selection and validation techniques using scikit-learn.

## Features

- K-Fold and Stratified K-Fold Cross-Validation
- Grid Search for Hyperparameter Tuning
- Learning Curve Visualization
- Model Performance Evaluation (Confusion Matrix, ROC Curve)
- Support for Multiple Models (Random Forest, Logistic Regression)

## Project Structure

```
day7_model_selection/
├── data/               # Data directory
├── src/               # Source code
│   ├── model_selector.py  # Main model selection class
│   └── main.py           # Example usage script
├── tests/             # Test files
├── output/            # Output directory for visualizations and results
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset in the `data` directory
2. Run the main script:
```bash
python src/main.py
```

## Output

The script generates several outputs in the `output` directory:
- `cv_results.csv`: Cross-validation results
- `grid_search_results.csv`: Grid search results
- `learning_curve.png`: Learning curve visualization
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve visualization

## Model Selection Process

1. Data Preparation
   - Load and split data
   - Handle class imbalance using stratified split

2. Cross-Validation
   - K-Fold and Stratified K-Fold
   - Calculate accuracy and F1 scores
   - Save results for analysis

3. Hyperparameter Tuning
   - Grid Search with cross-validation
   - Parameter grids for different models
   - Save best parameters and scores

4. Model Evaluation
   - Learning curves
   - Confusion matrices
   - ROC curves
   - Performance metrics

## Contributing

Feel free to submit issues and enhancement requests. 