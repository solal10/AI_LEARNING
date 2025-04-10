from setuptools import setup, find_packages

setup(
    name="california_housing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.0",
        "flake8>=3.9.0",
        "black>=21.7b0",
        "scipy>=1.7.0",
        "xgboost>=2.0.3",
    ],
    python_requires=">=3.9",
)
