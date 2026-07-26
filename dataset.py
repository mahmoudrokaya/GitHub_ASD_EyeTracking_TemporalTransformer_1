import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EyeTrackingDataset(Dataset):
    def __init__(self, csv_file, metadata_file, seq_len=30):
        df = pd.read_csv(csv_file, low_memory=False)
        meta = pd.read_csv(metadata_file)

        # Normalize participant IDs to integers
        df["Participant"] = pd.to_numeric(df["Participant"], errors="coerce")
        meta["ParticipantID"] = pd.to_numeric(meta["ParticipantID"], errors="coerce")

        # Drop any rows where participant ID is missing
        df = df.dropna(subset=["Participant"])
        meta = meta.dropna(subset=["ParticipantID"])

        label_map = {"ASD": 1, "TD": 0, "TS": 0, "TC": 0}
        meta["Class"] = meta["Class"].map(label_map)
        participant_labels = dict(zip(meta["ParticipantID"], meta["Class"]))

        df = df.replace("-", np.nan)
        df["Point of Regard Right X [px]"] = pd.to_numeric(df["Point of Regard Right X [px]"], errors='coerce')
        df["Point of Regard Right Y [px]"] = pd.to_numeric(df["Point of Regard Right Y [px]"], errors='coerce')
        df = df.dropna(subset=["Point of Regard Right X [px]", "Point of Regard Right Y [px]"])

        self.sequences = []
        self.labels = []

        grouped = df.groupby("Participant")
        print("Total participants in CSV:", len(grouped))

        for pid, data in grouped:
            if pid not in participant_labels:
                print(f"Skipping participant {pid}: not in metadata")
                continue

            coords = data[["Point of Regard Right X [px]", "Point of Regard Right Y [px]"]].values
            if len(coords) < seq_len:
                print(f"Skipping participant {pid}: only {len(coords)} valid rows")
                continue

            label = participant_labels[pid]
            for i in range(0, len(coords) - seq_len + 1):
                seq = coords[i:i + seq_len]
                self.sequences.append(seq)
                self.labels.append(label)

        print("Final valid sequences:", len(self.sequences))
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
