import os
import gc
import psutil
import pandas as pd

from src.extract import extract_data
from src.transform import transform_row
from src.load import load_data


def show_memory(stage):
    """
    Display the current memory usage of the Python process.
    """
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024)
    print(f"{stage}: {memory:.2f} MB")


# ----------------------------
# Memory Before ETL
# ----------------------------
show_memory("Memory Before ETL")


# ----------------------------
# Extract + Transform
# ----------------------------
cleaned_data = []

for row in extract_data("data/sample_data.csv"):
    cleaned_row = transform_row(row)
    cleaned_data.append(cleaned_row)


# ----------------------------
# Load
# ----------------------------
df = load_data(cleaned_data)


# ----------------------------
# Display Results
# ----------------------------
print("\nFirst 5 Rows:")
print(df.head())

print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ----------------------------
# Save Cleaned Data
# ----------------------------
df.to_csv("data/cleaned_data.csv", index=False)


# ----------------------------
# Memory After Generator ETL
# ----------------------------
show_memory("Memory After Generator ETL")


# ----------------------------
# DataFrame Memory Usage
# ----------------------------
print("\nDataFrame Memory Usage:")
print(df.memory_usage(deep=True))

total_memory = df.memory_usage(deep=True).sum() / 1024
print(f"\nTotal DataFrame Memory: {total_memory:.2f} KB")


# ----------------------------
# Compare with pd.read_csv()
# ----------------------------
pandas_df = pd.read_csv("data/sample_data.csv")

show_memory("Memory After pd.read_csv()")


# ----------------------------
# Garbage Collection
# ----------------------------
del cleaned_data
del df
del pandas_df

gc.collect()

show_memory("Memory After Garbage Collection")