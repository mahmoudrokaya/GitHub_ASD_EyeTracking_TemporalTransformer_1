import pandas as pd
import os
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
BASE_INPUT_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output"
METADATA_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1"
OUTPUT_FOLDER = os.path.join(BASE_INPUT_FOLDER, "Balanced_70_15_15")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
CLASS_COLUMN = "Class"
PARTICIPANT_COLUMN = "Participant"
META_PARTICIPANT_COLUMN = "value"

# === LOAD ALL FILES ===
def load_all_data():
    all_files = []
    for filename in os.listdir(BASE_INPUT_FOLDER):
        if filename.endswith(".csv"):
            filepath = os.path.join(BASE_INPUT_FOLDER, filename)
            try:
                df = pd.read_csv(filepath, low_memory=False)
                df["SourceFile"] = filename
                all_files.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return pd.concat(all_files, ignore_index=True)

# === SPLIT BY PARTICIPANT ===
def split_by_participant(data, participants_df):
    train_rows, val_rows, test_rows = [], [], []

    for _, row in participants_df.iterrows():
        participant_id = row[META_PARTICIPANT_COLUMN]
        subset = data[data[PARTICIPANT_COLUMN] == participant_id]

        if subset.empty:
            continue

        try:
            train_part, temp_part = train_test_split(
                subset, test_size=(1 - RATIOS['train']),
                stratify=subset[CLASS_COLUMN], random_state=42
            )
            val_part, test_part = train_test_split(
                temp_part, test_size=RATIOS['test'] / (RATIOS['val'] + RATIOS['test']),
                stratify=temp_part[CLASS_COLUMN], random_state=42
            )
        except Exception:
            # Fallback without stratify
            subset = subset.sample(frac=1, random_state=42)
            n = len(subset)
            n_train = int(n * RATIOS['train'])
            n_val = int(n * RATIOS['val'])
            train_part = subset.iloc[:n_train]
            val_part = subset.iloc[n_train:n_train + n_val]
            test_part = subset.iloc[n_train + n_val:]

        train_rows.append(train_part)
        val_rows.append(val_part)
        test_rows.append(test_part)

    return pd.concat(train_rows), pd.concat(val_rows), pd.concat(test_rows)

# === SAVE OUTPUT ===
def save_splits(train_df, val_df, test_df):
    train_df.to_csv(os.path.join(OUTPUT_FOLDER, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_FOLDER, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_FOLDER, "test.csv"), index=False)

    print("Split completed and saved.")
    print(f"Train set class counts: {train_df['Class'].value_counts().to_dict()}")
    print(f"Validation set class counts: {val_df['Class'].value_counts().to_dict()}")
    print(f"Test set class counts: {test_df['Class'].value_counts().to_dict()}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        full_data = load_all_data()
        participants_info = pd.read_csv(os.path.join(METADATA_FOLDER, "participant_value_frequencies.csv"))
        participants_info[META_PARTICIPANT_COLUMN] = participants_info[META_PARTICIPANT_COLUMN].astype(str)
        full_data[PARTICIPANT_COLUMN] = full_data[PARTICIPANT_COLUMN].astype(str)

        train, val, test = split_by_participant(full_data, participants_info)
        save_splits(train, val, test)
    except Exception as err:
        print("Error during processing:", str(err))
