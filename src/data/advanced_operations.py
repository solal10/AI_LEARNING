import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Create a DataFrame with feature names
df = pd.DataFrame(X, columns=housing.feature_names)
df["target"] = y  # Add the target variable (house prices)

# Rename columns to match the suggested naming convention
df = df.rename(
    columns={
        "target": "median_house_value",
        "MedInc": "median_income",
        "HouseAge": "housing_median_age",
        "Population": "population",
    }
)

# Add a categorical feature for demonstration (ocean_proximity)
np.random.seed(42)
ocean_proximity = np.random.choice(
    ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"], size=len(df)
)
df["ocean_proximity"] = ocean_proximity

# Add a datetime column for demonstration
base_date = datetime(2020, 1, 1)
dates = [base_date + timedelta(days=x) for x in range(len(df))]
df["listing_date"] = dates

# Add a synthetic area column (in square feet) for price per square foot calculation
np.random.seed(42)
df["area_sqft"] = np.random.randint(800, 4000, size=len(df))

print("Original DataFrame:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())

# 1. Advanced Aggregation - Multiple Variables
print("\n1. Advanced Aggregation - Multiple Variables:")
# Group by ocean_proximity and calculate statistics for multiple variables
advanced_grouped_stats = (
    df.groupby("ocean_proximity")
    .agg(
        {
            "median_house_value": ["mean", "max", "min"],
            "median_income": ["mean"],
            "housing_median_age": ["mean"],
        }
    )
    .round(2)
)
print(advanced_grouped_stats)

# 2. Create a new feature - price per person
print("\n2. Create a new feature - price per person:")
df["price_per_person"] = df["median_house_value"] * 100000 / df["population"]
print("Price per person (first 5 rows):")
print(df[["median_house_value", "population", "price_per_person"]].head())

# 3. Create a column with apply() + lambda
print("\n3. Create a column with apply() + lambda:")
# Create a 'haut_de_gamme' (high-end) label based on median_house_value
df["haut_de_gamme"] = df["median_house_value"].apply(
    lambda x: "Oui" if x > 3.5 else "Non"
)
print("High-end label (first 5 rows):")
print(df[["median_house_value", "haut_de_gamme"]].head())

# 4. Advanced Pivot Table
print("\n4. Advanced Pivot Table:")
# Create a pivot table showing average median income by ocean proximity and high-end category
advanced_pivot = pd.pivot_table(
    df,
    values="median_income",
    index="ocean_proximity",
    columns="haut_de_gamme",
    aggfunc="mean",
    fill_value=0,
).round(2)
print(advanced_pivot)

# 5. Merge/Join/Concat - Simulate two DataFrames
print("\n5. Merge/Join/Concat - Simulate two DataFrames:")
# Create two separate DataFrames with a common ID
df["id"] = range(1, len(df) + 1)  # Add an ID column

# Create df_prix with price information
df_prix = df[["id", "median_house_value", "price_per_person", "haut_de_gamme"]].copy()

# Create df_info with property information
df_info = df[
    [
        "id",
        "ocean_proximity",
        "housing_median_age",
        "AveRooms",
        "AveBedrms",
        "population",
    ]
].copy()

# Merge the DataFrames
merged_df = pd.merge(df_prix, df_info, on="id", how="inner")
print("Merged DataFrame (first 5 rows):")
print(merged_df.head())

# 6. Additional Analysis - Correlation
print("\n6. Correlation Analysis:")
# Calculate correlation between numeric columns
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
correlation = df[numeric_cols].corr().round(2)
print(correlation)

# 7. Advanced GroupBy with multiple operations
print("\n7. Advanced GroupBy with multiple operations:")
# Group by ocean_proximity and haut_de_gamme, then calculate multiple statistics
advanced_groupby = (
    df.groupby(["ocean_proximity", "haut_de_gamme"])
    .agg(
        {
            "median_house_value": ["count", "mean", "std"],
            "median_income": ["mean", "min", "max"],
            "housing_median_age": "mean",
        }
    )
    .round(2)
)
print(advanced_groupby)

# 8. Time-based analysis
print("\n8. Time-based analysis:")
# Extract month from listing_date
df["month"] = df["listing_date"].dt.month

# Group by month and haut_de_gamme
time_analysis = (
    df.groupby(["month", "haut_de_gamme"])["median_house_value"]
    .mean()
    .unstack()
    .round(2)
)
print(time_analysis)

# Visualizations
plt.figure(figsize=(15, 12))

# 1. Bar chart of average median income by ocean proximity and high-end category
plt.subplot(2, 2, 1)
advanced_pivot.plot(kind="bar")
plt.title("Average Median Income by Ocean Proximity and High-End Category")
plt.ylabel("Median Income")
plt.xlabel("Ocean Proximity")
plt.legend(title="High-End")
plt.xticks(rotation=45)

# 2. Box plot of price per person by ocean proximity
plt.subplot(2, 2, 2)
df.boxplot(column="price_per_person", by="ocean_proximity")
plt.title("Price per Person by Ocean Proximity")
plt.ylabel("Price per Person ($)")
plt.xticks(rotation=45)

# 3. Heatmap of correlation
plt.subplot(2, 2, 3)
plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)

# 4. Time series of high-end vs non-high-end properties
plt.subplot(2, 2, 4)
time_analysis.plot(kind="line", marker="o")
plt.title("Average House Value by Month and High-End Category")
plt.ylabel("Average House Value (in $100,000s)")
plt.xlabel("Month")
plt.grid(True)
plt.legend(title="High-End")

plt.tight_layout()
plt.show()

# Save the processed DataFrame to CSV
df.to_csv("california_housing_advanced.csv", index=False)
print("\nProcessed DataFrame saved to 'california_housing_advanced.csv'")
