import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset1 import EyeTrackingDataset
from model1 import TemporalTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

INPUT_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output\Balanced_70_15_15"
META_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Metadata_Participants.csv"

EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
SEQ_LEN = 30
MAX_SEQ_PER_PARTICIPANT = 100
PATIENCE = 5

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer(dropout=0.4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_set = EyeTrackingDataset(os.path.join(INPUT_FOLDER, "train.csv"), META_PATH, seq_len=SEQ_LEN, max_seq_per_participant=MAX_SEQ_PER_PARTICIPANT)
    val_set = EyeTrackingDataset(os.path.join(INPUT_FOLDER, "val.csv"), META_PATH, seq_len=SEQ_LEN, max_seq_per_participant=MAX_SEQ_PER_PARTICIPANT)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

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
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Val Loss {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(INPUT_FOLDER, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(INPUT_FOLDER, "loss_curve.png"))

# Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(INPUT_FOLDER, "best_model.pt")))
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    # Confusion Matrix Figure
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["TD", "ASD"], yticklabels=["TD", "ASD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(INPUT_FOLDER, "confusion_matrix.png"))
    plt.close()

    # Classification Report Bar Plots
    report = classification_report(all_labels, all_preds, target_names=["TD", "ASD"], output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    for metric in metrics:
        values = [report["TD"][metric], report["ASD"][metric]]
        plt.figure()
        sns.barplot(x=["TD", "ASD"], y=values)
        plt.ylim(0, 1)
        plt.title(f"{metric.capitalize()} per Class")
        plt.ylabel(metric.capitalize())
        plt.savefig(os.path.join(INPUT_FOLDER, f"{metric}_barplot.png"))
        plt.close()

    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(INPUT_FOLDER, "roc_curve.png"))
        plt.close()
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        print("ROC AUC could not be computed (possibly only one class present).")

if __name__ == "__main__":
    train_model()

