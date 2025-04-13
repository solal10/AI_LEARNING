from setuptools import setup, find_packages

setup(
    name="ai_learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'seaborn>=0.11.0',
        'matplotlib>=3.4.0',
        'joblib>=1.0.0',
        'pyarrow>=14.0.0',
    ],
)
