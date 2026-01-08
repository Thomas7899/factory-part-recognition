# src/train.py

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights

import numpy as np

# -----------------------------------------
# Matplotlib: NON-BLOCKING (sehr wichtig!)
# -----------------------------------------
import matplotlib
matplotlib.use("Agg")  # verhindert Fenster & Blockieren
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =========================================
# Projektpfade
# =========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "car-parts-50"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# =========================================
# 1) Hyperparameter & Setup
# =========================================
epochs = 5
batch_size = 32
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"➡️ Using device: {device}")


# =========================================
# 2) Data Transforms
# =========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================================
# 3) Datasets & DataLoader
# =========================================
train_dataset = ImageFolder(str(DATA_DIR / "train"), transform=transform)
val_dataset = ImageFolder(str(DATA_DIR / "val"), transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

classes = train_dataset.classes
num_classes = len(classes)

print(f"➡️ Classes ({num_classes}): {classes}")


# =========================================
# 4) Modell, Loss, Optimizer
# =========================================
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Letzte Schicht anpassen
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =========================================
# 5) Evaluation-Funktion
# =========================================
def evaluate(model, dataloader, epoch: int):
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    wrong_images = []
    wrong_preds = []
    wrong_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    wrong_images.append(images[i].cpu())
                    wrong_preds.append(predicted[i].cpu())
                    wrong_labels.append(labels[i].cpu())

    # -------------------------
    # Confusion Matrix speichern
    # -------------------------
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90)
    plt.title(f"Confusion Matrix – Epoch {epoch+1}")

    cm_path = PLOTS_DIR / f"confusion_matrix_epoch_{epoch+1}.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()

    # -------------------------
    # Falsch klassifizierte Bilder
    # -------------------------
    if wrong_images:
        n = min(5, len(wrong_images))
        fig, axes = plt.subplots(1, n, figsize=(15, 4))

        if n == 1:
            axes = [axes]

        for ax, img, pred, true in zip(
            axes, wrong_images[:n], wrong_preds[:n], wrong_labels[:n]
        ):
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            ax.set_title(f"T:{classes[true]} | P:{classes[pred]}")
            ax.axis("off")

        wrong_path = PLOTS_DIR / f"wrong_predictions_epoch_{epoch+1}.png"
        plt.savefig(wrong_path, bbox_inches="tight", dpi=150)
        plt.close()

    model.train()
    return correct / total


# =========================================
# 6) Training Loop
# =========================================
print("🚀 Training startet...\n")

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    val_acc = evaluate(model, val_loader, epoch)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Acc: {val_acc:.2%}"
    )


# =========================================
# 7) Modell speichern
# =========================================
model_path = MODELS_DIR / "factory_cnn.pt"
torch.save(model.state_dict(), model_path)

print("\n✅ Training abgeschlossen!")
print(f"📦 Modell gespeichert unter: {model_path}")
print(f"📊 Plots gespeichert unter: {PLOTS_DIR}")
