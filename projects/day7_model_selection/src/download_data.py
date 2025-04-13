import os
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """Download the heart disease dataset from Kaggle."""
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        print("Downloading dataset from Kaggle...")
        # Download dataset
        api.dataset_download_files(
            'fedesoriano/heart-failure-prediction',
            path="data",
            unzip=True
        )
        
        print("Processing dataset...")
        # Read and prepare the data
        df = pd.read_csv("data/heart.csv")
        
        # Save the prepared data
        df.to_csv("data/heart_disease.csv", index=False)
        print("Dataset downloaded and prepared successfully!")
        
        # Remove the original file
        os.remove("data/heart.csv")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_dataset() 