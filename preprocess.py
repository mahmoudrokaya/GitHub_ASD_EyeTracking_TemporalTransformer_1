import pandas as pd
import os
from sklearn.model_selection import train_test_split

BASE_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\EX1"
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "Balanced")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_existing_sets():
    try:
        train = pd.read_csv(os.path.join(BASE_FOLDER, "train.csv"), low_memory=False)
        val = pd.read_csv(os.path.join(BASE_FOLDER, "val.csv"), low_memory=False)
        test = pd.read_csv(os.path.join(BASE_FOLDER, "test.csv"), low_memory=False)
        unidentified = pd.read_csv(os.path.join(BASE_FOLDER, "unidentified.csv"), low_memory=False)
        all_data = pd.concat([train, val, test], ignore_index=True)
        return all_data, unidentified
    except Exception as e:
        print("Error loading original splits:", e)
        exit(1)

def fallback_balanced_split(participants):
    classes = participants["Class"].unique()
    splits = {"train": [], "val": [], "test": []}

    for c in classes:
        subset = participants[participants["Class"] == c]
        if len(subset) < 3:
            print(f"Class '{c}' has fewer than 3 participants. Distributing to train only.")
            splits["train"].append(subset)
        else:
            train_c, temp_c = train_test_split(subset, test_size=0.4, random_state=42)
            val_c, test_c = train_test_split(temp_c, test_size=0.5, random_state=42)
            splits["train"].append(train_c)
            splits["val"].append(val_c)
            splits["test"].append(test_c)

    return pd.concat(splits["train"]), pd.concat(splits["val"]), pd.concat(splits["test"])

def safe_stratified_split(participants):
    try:
        train_part, temp_part = train_test_split(
            participants,
            test_size=0.4,
            stratify=participants["Class"],
            random_state=42
        )
        val_part, test_part = train_test_split(
            temp_part,
            test_size=0.5,
            stratify=temp_part["Class"],
            random_state=42
        )
        return train_part, val_part, test_part
    except Exception as e:
        print("Stratified split failed, using manual balancing. Reason:", str(e))
        return fallback_balanced_split(participants)

def save_split_data(data, train_part, val_part, test_part, unidentified):
    def get_data(part_ids):
        return data[data["Participant"].isin(part_ids["Participant"])]

    for name, part in zip(["train", "val", "test"], [train_part, val_part, test_part]):
        split_df = get_data(part)
        split_df.to_csv(os.path.join(OUTPUT_FOLDER, f"{name}.csv"), index=False)
        print(f"{name.title()} set class counts: {split_df['Class'].value_counts().to_dict()}")

    unidentified.to_csv(os.path.join(OUTPUT_FOLDER, "unidentified.csv"), index=False)
    print(f"Unidentified records saved: {len(unidentified)}")

if __name__ == "__main__":
    all_data, unidentified_df = load_existing_sets()

    # Only use rows with known labels (ASD or TD)
    labeled_data = all_data[all_data["Class"].isin(["ASD", "TD"])]
    participant_ids = labeled_data[["Participant", "Class"]].drop_duplicates()

    train_p, val_p, test_p = safe_stratified_split(participant_ids)
    save_split_data(all_data, train_p, val_p, test_p, unidentified_df)
    print(f"Rebalanced data saved to: {OUTPUT_FOLDER}")
