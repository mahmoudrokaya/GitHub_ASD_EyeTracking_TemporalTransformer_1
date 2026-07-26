import pandas as pd

CSV_FILE = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output\Balanced_70_15_15\train.csv"
SEQ_LEN = 100
X_COL = "Point of Regard Right X [px]"
Y_COL = "Point of Regard Right Y [px]"
CLASS_COL = "groupe d'enfants"  # This should match exactly

df = pd.read_csv(CSV_FILE, low_memory=False)

print(f"Initial rows: {len(df)}")

# Rename for consistency
df.rename(columns={df.columns[7]: "Participant", df.columns[58]: "Class"}, inplace=True)
df["Class"] = df["Class"].replace({"TS": "TD", "TC": "TD"})

label_map = {"TD": 0, "ASD": 1}
df = df[df["Class"].isin(label_map.keys())]

# Remove invalid gaze rows
df = df[(df[X_COL] != '-') & (df[Y_COL] != '-')]
df[X_COL] = df[X_COL].astype(float)
df[Y_COL] = df[Y_COL].astype(float)

valid_sequences = 0
for pid, data in df.groupby("Participant"):
    if len(data) >= SEQ_LEN:
        sequences = len(data) // SEQ_LEN
        valid_sequences += sequences
        print(f"Participant: {pid} → sequences: {sequences}")

print(f"Total valid sequences: {valid_sequences}")
