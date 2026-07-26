import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# === PATHS ===
metadata_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\Eye-Tracking Dataset\Metadata_Participants.csv"
data_dir = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\Eye-Tracking Dataset\Eye-tracking Output"
output_base = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\Eye-Tracking Dataset\Eye-tracking Output\Training&Val&Testing_Data"

# === OUTPUT FOLDERS ===
os.makedirs(os.path.join(output_base, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'test'), exist_ok=True)

# === Load metadata ===
metadata = pd.read_csv(metadata_path)
metadata['ParticipantID'] = metadata['ParticipantID'].astype(str)
metadata_map = dict(zip(metadata['ParticipantID'], metadata['Class']))

# === Collect valid files ===
file_records = []
for i in range(1, 26):
    fname = f"{i}.csv"
    fpath = os.path.join(data_dir, fname)
    try:
        df = pd.read_csv(fpath, nrows=1)
        pid = str(df['Participant'].iloc[0])
        if pid.startswith("Unidentified"):
            continue  # skip unidentified
        if pid in metadata_map:
            file_records.append({
                'filename': fname,
                'participant': pid,
                'class': metadata_map[pid]
            })
    except Exception as e:
        print(f"Skipped {fname}: {e}")

df_files = pd.DataFrame(file_records)
print(f"\nIdentified files: {len(df_files)}")
print(df_files[['filename', 'participant', 'class']])

# === Stratified Splits ===
train_val, test = train_test_split(
    df_files, test_size=0.2, stratify=df_files['class'], random_state=42)

class_counts = train_val['class'].value_counts()
if (class_counts < 2).any():
    print("Not enough samples per class for stratified validation split. Using random split instead.")
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)
else:
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['class'], random_state=42)

# === Helper to copy files ===
def copy_files(subset_df, out_folder):
    for _, row in subset_df.iterrows():
        src = os.path.join(data_dir, row['filename'])
        dst = os.path.join(out_folder, row['filename'])
        shutil.copyfile(src, dst)

copy_files(train, os.path.join(output_base, 'train'))
copy_files(val, os.path.join(output_base, 'val'))
copy_files(test, os.path.join(output_base, 'test'))

# === Print class distribution ===
def print_class_distribution(df, name):
    print(f"\n{name} set class distribution:")
    print(df['class'].value_counts())

print_class_distribution(train, "Train")
print_class_distribution(val, "Validation")
print_class_distribution(test, "Test")

print("\nData splitting and file copying completed.")
