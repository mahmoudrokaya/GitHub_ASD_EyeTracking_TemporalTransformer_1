import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EyeTrackingDataset
from model import TemporalTransformer
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import os

INPUT_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output\Balanced_70_15_15"
META_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Metadata_Participants.csv"

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
SEQ_LEN = 30

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_set = EyeTrackingDataset(os.path.join(INPUT_FOLDER, "train.csv"), META_PATH, seq_len=SEQ_LEN)
    val_set = EyeTrackingDataset(os.path.join(INPUT_FOLDER, "val.csv"), META_PATH, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item()
                all_preds.extend(out.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}")

    torch.save(model.state_dict(), os.path.join(INPUT_FOLDER, "model.pt"))

    # Plot Loss
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(INPUT_FOLDER, "loss_curve.png"))

if __name__ == "__main__":
    train_model()
