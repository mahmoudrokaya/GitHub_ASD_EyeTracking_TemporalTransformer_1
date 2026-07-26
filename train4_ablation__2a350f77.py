import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset1_variant import EyeTrackingDatasetVariant
from model4_ablation import TemporalTransformerAblation
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

INPUT_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output\Balanced_70_15_15"
META_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Metadata_Participants.csv"
OUTPUT_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Experiment4\AblationResults"

SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
PATIENCE = 3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformerAblation().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_set = EyeTrackingDatasetVariant(os.path.join(INPUT_FOLDER, "train.csv"), META_PATH, seq_len=SEQ_LEN)
    val_set = EyeTrackingDatasetVariant(os.path.join(INPUT_FOLDER, "val.csv"), META_PATH, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    best_loss = float('inf')
    patience = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss, preds, labels = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item()
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, "best_model.pt"))
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    # Final Evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, "best_model.pt")))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            all_probs.extend(out.softmax(dim=1)[:, 1].cpu().numpy())
            all_labels.extend(y.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    print("ROC AUC:", auc)

if __name__ == "__main__":
    train_model()
