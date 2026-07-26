import pandas as pd
import numpy as np

csv_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output\Balanced_70_15_15\train.csv"
df = pd.read_csv(csv_path, low_memory=False)

print("Raw rows:", df.shape[0])
print("Columns:", list(df.columns))

# Check for common issues
print("\nUnique values in 'groupe d'enfants':", df["groupe d'enfants"].unique())
print("Number of missing in Point of Regard Right X [px]:", df["Point of Regard Right X [px]"].isna().sum())
print("Number of '-' values in X:", (df["Point of Regard Right X [px]"] == "-").sum())

# Try cleaning
df[["Point of Regard Right X [px]", "Point of Regard Right Y [px]"]] = df[
    ["Point of Regard Right X [px]", "Point of Regard Right Y [px]"]
].replace("-", np.nan).astype(float)

valid = df.dropna(subset=["Point of Regard Right X [px]", "Point of Regard Right Y [px]"])
print("Valid gaze rows after cleaning:", valid.shape[0])
print("Valid unique participants:", valid["Participant"].nunique())
