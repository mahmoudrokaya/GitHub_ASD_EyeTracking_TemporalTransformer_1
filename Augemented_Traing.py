import pandas as pd
from ctgan import CTGAN
import os

# Define paths
base_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\EX1"
input_csv = os.path.join(base_path, "Train.csv")
output_csv = os.path.join(base_path, "Synthetic_Data.csv")

# Load the dataset with dtype warning suppression
data = pd.read_csv(input_csv, low_memory=False)

# Drop columns with high cardinality or too many missing values
high_cardinality_cols = [col for col in data.columns if data[col].nunique() > 1000]
many_missing_cols = [col for col in data.columns if data[col].isnull().sum() > len(data) * 0.5]
data.drop(columns=high_cardinality_cols + many_missing_cols, inplace=True)

# Fill missing values to prevent encoding issues
data.fillna("missing", inplace=True)

# Identify categorical columns (low cardinality or string type)
categorical_columns = [
    col for col in data.columns
    if data[col].dtype == 'object' or data[col].nunique() < 20
]

# Initialize and train the CTGAN model
model = CTGAN(epochs=100)
model.fit(data, discrete_columns=categorical_columns)

# Generate synthetic data
synthetic_data = model.sample(len(data))

# Save synthetic dataset
synthetic_data.to_csv(output_csv, index=False)
print(f"Synthetic data saved to: {output_csv}")
