import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
import seaborn as sns

# ==== CONFIGURATION ====
INPUT_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Experiment5\Inputs"
OUTPUT_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Experiment5\Outputs"
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== CUSTOM DATASET ====
class GazeDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# ==== MODEL ====
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=128, embed_dim=64, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x

# ==== LOAD DATA ====
train_dataset = GazeDataset(os.path.join(INPUT_PATH, "train.csv"))
val_dataset = GazeDataset(os.path.join(INPUT_PATH, "val.csv"))
test_dataset = GazeDataset(os.path.join(INPUT_PATH, "test.csv"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==== TRAINING FUNCTION ====
def train_model():
    model = TemporalTransformer(input_dim=train_dataset[0][0].shape[0]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.unsqueeze(1)  # add temporal dimension
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.unsqueeze(1)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "model.pt"))

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, "Loss_Curve.png"))
    plt.close()

    return model

# ==== EVALUATION FUNCTION ====
def evaluate_model(model):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE).unsqueeze(1)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=["TD", "ASD"], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(OUTPUT_PATH, "Classification_Report.csv"))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["TD", "ASD"], yticklabels=["TD", "ASD"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_PATH, "Confusion_Matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, "ROC_Curve.png"))
    plt.close()

    print(f"Evaluation complete. AUC: {auc:.3f}")

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model = train_model()
    evaluate_model(model)
