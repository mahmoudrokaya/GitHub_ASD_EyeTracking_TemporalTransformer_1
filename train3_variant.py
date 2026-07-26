import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset1 import EyeTrackingDataset
from model3_variant import TemporalTransformerVariant
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== CONFIGURATION ====
INPUT_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output\Balanced_70_15_15"
META_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Metadata_Participants.csv"
OUTPUT_FOLDER = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Experiment3\VariantResults"

INPUT_DIM = 2  # FIXED to match actual input feature shape
SEQ_LEN = 30
MAX_SEQ_PER_PARTICIPANT = 100
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== TRAINING FUNCTION ====
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformerVariant(input_dim=INPUT_DIM, dropout=0.4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_set = EyeTrackingDataset(os.path.join(INPUT_FOLDER, "train.csv"), META_PATH, seq_len=SEQ_LEN, max_seq_per_participant=MAX_SEQ_PER_PARTICIPANT)
    val_set = EyeTrackingDataset(os.path.join(INPUT_FOLDER, "val.csv"), META_PATH, seq_len=SEQ_LEN, max_seq_per_participant=MAX_SEQ_PER_PARTICIPANT)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

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
            torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # ==== LOSS CURVE ====
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "loss_curve.png"))

    # ==== FINAL EVALUATION ====
    print("\nFinal Evaluation on Validation Set:")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, "best_model.pt")))
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            all_probs.extend(out.softmax(dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    report = classification_report(all_labels, all_preds, output_dict=True)
    roc_auc = roc_auc_score(all_labels, all_probs)
    print("ROC AUC Score:", roc_auc)

    # ==== METRIC PLOTS ====
    for metric in ["precision", "recall", "f1-score"]:
        values = [report[str(i)][metric] for i in range(2)]
        plt.figure()
        sns.barplot(x=["Class 0", "Class 1"], y=values)
        plt.title(f"{metric.title()} by Class")
        plt.ylabel(metric.title())
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"{metric}_barplot.png"))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"))

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "roc_curve.png"))

if __name__ == "__main__":
    train_model()
