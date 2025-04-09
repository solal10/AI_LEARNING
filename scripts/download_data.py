import os
from sklearn.datasets import fetch_california_housing
import pandas as pd


def download_california_housing():
    """Download the California Housing dataset and save it as CSV."""
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    # Load the dataset
    california = fetch_california_housing()

    # Create a DataFrame with correct column names
    column_mapping = {
        "MedInc": "median_income",
        "HouseAge": "housing_median_age",
        "AveRooms": "AveRooms",
        "AveBedrms": "AveBedrms",
        "Population": "population",
        "AveOccup": "AveOccup",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
    }

    # Create DataFrame with original column names
    df = pd.DataFrame(california.data, columns=california.feature_names)

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Add target variable
    df["median_house_value"] = california.target

    # Add ocean_proximity as a random categorical variable for demonstration
    # In a real dataset, this would be actual data
    df["ocean_proximity"] = (
        pd.Series(["NEAR BAY", "INLAND", "<1H OCEAN", "NEAR OCEAN", "ISLAND"])
        .sample(n=len(df), replace=True, random_state=42)
        .values
    )

    # Save to CSV
    output_file = "data/raw/california_housing.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Shape: {df.shape}")
    print("\nFeatures:")
    print(df.columns.tolist())


if __name__ == "__main__":
    download_california_housing()
