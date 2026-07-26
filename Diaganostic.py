import pandas as pd
import numpy as np

df = pd.read_csv("E:/Mahmoud/Exams/46/462/New-papers/Paper6/NewDataSets/Data1/Eye-tracking Output/Balanced_70_15_15/train.csv", low_memory=False)

# Replace '-' with NaN for coordinate columns
df[["Point of Regard Right X [px]", "Point of Regard Right Y [px]"]] = df[
    ["Point of Regard Right X [px]", "Point of Regard Right Y [px]"]
].replace('-', np.nan).astype(float)

valid = df.dropna(subset=["Point of Regard Right X [px]", "Point of Regard Right Y [px]"])
print("Total participants:", df["Participant"].nunique())
print("Valid rows after cleaning:", valid.shape[0])
