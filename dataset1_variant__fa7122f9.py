import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class EyeTrackingDataset(Dataset):
    def __init__(self, csv_file, metadata_file, seq_len=30, max_seq_per_participant=100):
        self.seq_len = seq_len
        self.max_seq_per_participant = max_seq_per_participant

        df = pd.read_csv(csv_file)
        meta = pd.read_csv(metadata_file)

        df = df.replace("-", np.nan)
        df = df.dropna()
        df['Participant'] = df['Participant'].astype(int)

        # Map participant to label using metadata
        participant_labels = meta.set_index('ParticipantID')['Class'].map(lambda x: 1 if str(x).strip().upper() == 'ASD' else 0)
        df['label'] = df['Participant'].map(participant_labels)

        # Remove entries with missing label
        df = df[df['label'].notna()]

        # Group and prepare sequences
        self.samples = []
        for pid, group in df.groupby('Participant'):
            features = group[
            ['Gaze Vector Left X', 'Gaze Vector Left Y', 'Gaze Vector Left Z',
            'Gaze Vector Right X', 'Gaze Vector Right Y', 'Gaze Vector Right Z']
                ].astype(float).to_numpy()

            for i in range(0, min(len(features) - seq_len + 1, self.max_seq_per_participant)):
                seq = features[i:i + seq_len]
                label = group['label'].iloc[0]
                self.samples.append((seq, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
