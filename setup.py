from setuptools import setup, find_packages

setup(
    name="california_housing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
        "matplotlib",
        "seaborn"
    ],
) 