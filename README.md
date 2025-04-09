# California Housing Price Prediction

[![CI/CD Pipeline](https://github.com/yourusername/california-housing/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/california-housing/actions/workflows/ci.yml)

A machine learning pipeline for analyzing and predicting California housing prices.

## Description

This project implements a complete machine learning pipeline for the California Housing dataset, including:
- Data loading and validation
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Report generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/california-housing.git
cd california-housing
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

Run the pipeline with:
```bash
python -m src.main --input_file data/raw/california_housing.csv
```

## Project Structure

```
california-housing/
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── src/
│   ├── models/        # Model implementations
│   ├── utils/         # Utility functions
│   └── reporting/     # Report generation
├── tests/             # Test files
├── requirements.txt   # Production dependencies
└── setup.py          # Package configuration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 