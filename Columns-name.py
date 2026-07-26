import pandas as pd

csv_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output\Balanced_70_15_15\train.csv"

# Load only the header row
df = pd.read_csv(csv_path, nrows=0)

# Print the column names
print("Column names:", list(df.columns))