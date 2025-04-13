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

# Add a categorical feature for demonstration (ocean_proximity)
# Since the original dataset doesn't have this, we'll create a synthetic one
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

# 1. GroupBy and Aggregation
print("\n1. GroupBy and Aggregation:")
# Group by ocean_proximity and calculate statistics
grouped_stats = (
    df.groupby("ocean_proximity")
    .agg(
        {
            "target": ["mean", "std", "min", "max", "count"],
            "MedInc": "mean",
            "area_sqft": "mean",
        }
    )
    .round(2)
)
print(grouped_stats)

# 2. Apply and Lambda
print("\n2. Apply and Lambda:")
# Calculate price per square foot
df["price_per_sqft"] = df.apply(
    lambda row: row["target"] * 100000 / row["area_sqft"], axis=1
)
print("Price per square foot (first 5 rows):")
print(df[["target", "area_sqft", "price_per_sqft"]].head())

# 3. Pivot Table
print("\n3. Pivot Table:")
# Create a pivot table showing average house price by ocean proximity and house age groups
df["house_age_group"] = pd.cut(
    df["HouseAge"], bins=[0, 10, 20, 30, 50], labels=["0-10", "11-20", "21-30", "31-50"]
)
pivot_table = pd.pivot_table(
    df,
    values="target",
    index="ocean_proximity",
    columns="house_age_group",
    aggfunc="mean",
    fill_value=0,
).round(2)
print(pivot_table)

# 4. Merge/Join/Concat
print("\n4. Merge/Join/Concat:")
# Create two separate DataFrames
df1 = df[["MedInc", "HouseAge", "target", "ocean_proximity"]].copy()
df2 = df[["AveRooms", "AveBedrms", "Population", "target", "ocean_proximity"]].copy()

# Merge the DataFrames
merged_df = pd.merge(df1, df2, on=["target", "ocean_proximity"], how="inner")
print("Merged DataFrame (first 5 rows):")
print(merged_df.head())

# 5. Datetime Operations
print("\n5. Datetime Operations:")
# Extract date components
df["year"] = df["listing_date"].dt.year
df["month"] = df["listing_date"].dt.month
df["day_of_week"] = df["listing_date"].dt.day_name()

# Group by month and calculate average price
monthly_avg_price = df.groupby("month")["target"].mean().round(2)
print("Average house price by month:")
print(monthly_avg_price)

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Bar chart of average price by ocean proximity
plt.subplot(2, 2, 1)
df.groupby("ocean_proximity")["target"].mean().plot(kind="bar")
plt.title("Average House Price by Ocean Proximity")
plt.ylabel("Average Price (in $100,000s)")
plt.xticks(rotation=45)

# 2. Box plot of price per square foot by ocean proximity
plt.subplot(2, 2, 2)
df.boxplot(column="price_per_sqft", by="ocean_proximity")
plt.title("Price per Square Foot by Ocean Proximity")
plt.ylabel("Price per Square Foot ($)")
plt.xticks(rotation=45)

# 3. Line chart of average price by month
plt.subplot(2, 2, 3)
monthly_avg_price.plot(kind="line", marker="o")
plt.title("Average House Price by Month")
plt.ylabel("Average Price (in $100,000s)")
plt.grid(True)

# 4. Scatter plot of price vs. median income, colored by ocean proximity
plt.subplot(2, 2, 4)
for proximity in df["ocean_proximity"].unique():
    subset = df[df["ocean_proximity"] == proximity]
    plt.scatter(subset["MedInc"], subset["target"], label=proximity, alpha=0.6)
plt.title("House Price vs. Median Income by Ocean Proximity")
plt.xlabel("Median Income")
plt.ylabel("House Price (in $100,000s)")
plt.legend()

plt.tight_layout()
plt.show()

# Save the processed DataFrame to CSV
df.to_csv("california_housing_processed.csv", index=False)
print("\nProcessed DataFrame saved to 'california_housing_processed.csv'")
