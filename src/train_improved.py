"""
Verbesserter Training-Pipeline mit:
- Config-basiert & reproduzierbar
- Professionelle Augmentations
- Early Stopping
- Learning Rate Scheduling
- Bessere Metriken
- Model Checkpointing
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import numpy as np

# Matplotlib Backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Lokale Imports
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config, get_device, set_seed, Config
from augmentations import get_train_transforms, get_val_transforms
from model import FactoryPartClassifier, save_model, count_parameters
from metrics import full_evaluation, generate_evaluation_report


# =========================================
# Projektpfade
# =========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

for dir_path in [MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)


class EarlyStopping:
    """Early Stopping zur Vermeidung von Overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class TrainingLogger:
    """Logging für Training Metriken."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rate": [],
        }
        self.start_time = datetime.now()
    
    def log(self, epoch: int, metrics: dict):
        """Loggt Metriken für eine Epoche."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Konsolen-Output
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {metrics.get('train_loss', 0):.4f} | "
            f"Val Acc: {metrics.get('val_accuracy', 0):.2%} | "
            f"Val F1: {metrics.get('val_f1', 0):.2%} | "
            f"LR: {metrics.get('learning_rate', 0):.2e}"
        )
    
    def save(self, filename: str = "training_history.json"):
        """Speichert Training History."""
        history_path = self.log_dir / filename
        
        save_data = {
            "history": self.history,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "timestamp": self.start_time.isoformat(),
        }
        
        with open(history_path, "w") as f:
            json.dump(save_data, f, indent=2)
        
        print(f"📊 Training History gespeichert: {history_path}")
    
    def plot_curves(self, save_path: Path):
        """Erstellt Training Curves Plot."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        if self.history["val_loss"]:
            axes[0].plot(self.history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history["val_accuracy"], label="Val Accuracy", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[2].plot(self.history["learning_rate"], label="Learning Rate", color="orange")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].set_yscale("log")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"📈 Training Curves gespeichert: {save_path}")


def create_dataloaders(config: Config) -> tuple:
    """Erstellt DataLoaders mit Augmentations."""
    
    data_dir = PROJECT_ROOT / config.data.root_dir
    
    # Transforms
    train_transform = get_train_transforms(
        img_size=config.data.img_size,
        config=config.augmentation.train
    )
    val_transform = get_val_transforms(img_size=config.data.img_size)
    
    # Datasets
    train_dataset = ImageFolder(
        str(data_dir / config.data.train_dir),
        transform=train_transform
    )
    val_dataset = ImageFolder(
        str(data_dir / config.data.val_dir),
        transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, train_dataset.classes


def create_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """Erstellt Optimizer basierend auf Config."""
    
    if config.training.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay,
        )
    else:
        raise ValueError(f"Unbekannter Optimizer: {config.training.optimizer}")


def create_scheduler(optimizer: optim.Optimizer, config: Config):
    """Erstellt Learning Rate Scheduler."""
    
    params = config.training.scheduler_params
    
    if config.training.scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("step_size", 5),
            gamma=params.get("gamma", 0.1),
        )
    elif config.training.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
        )
    elif config.training.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=params.get("patience", 3),
            factor=params.get("gamma", 0.1),
        )
    else:
        return None


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list,
) -> dict:
    """Evaluiert Modell und gibt Metriken zurück."""
    
    model.eval()
    
    all_probs = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Volle Evaluierung mit unseren Metriken
    metrics = full_evaluation(
        probs=all_probs,
        labels=all_labels,
        class_names=class_names,
    )
    
    return {
        "val_loss": total_loss / len(dataloader.dataset),
        "val_accuracy": metrics.accuracy,
        "val_f1": metrics.macro_f1,
        "metrics": metrics,
        "probs": all_probs,
        "labels": all_labels,
    }


def save_confusion_matrix(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str = "Confusion Matrix",
):
    """Speichert Confusion Matrix als Bild."""
    
    predictions = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, predictions)
    
    # Normalisierte Version
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Absolute Zahlen
    disp1 = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp1.plot(ax=axes[0], cmap=plt.cm.Blues, xticks_rotation=90)
    axes[0].set_title(f"{title} (Absolute)")
    
    # Normalisiert
    disp2 = ConfusionMatrixDisplay(cm_normalized, display_labels=class_names)
    disp2.plot(ax=axes[1], cmap=plt.cm.Blues, xticks_rotation=90, values_format=".2f")
    axes[1].set_title(f"{title} (Normalized)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def train(config_path: Path = None):
    """Haupttraining-Funktion."""
    
    # Config laden
    config = load_config(config_path)
    
    # Reproduzierbarkeit
    set_seed(config.seed, config.deterministic)
    
    # Device
    device = get_device()
    print(f"🖥️ Device: {device}")
    
    # Data
    print("📂 Lade Daten...")
    train_loader, val_loader, class_names = create_dataloaders(config)
    print(f"   Train: {len(train_loader.dataset)} Samples")
    print(f"   Val: {len(val_loader.dataset)} Samples")
    print(f"   Klassen: {len(class_names)}")
    
    # Modell
    print(f"\n🧠 Erstelle Modell: {config.model.architecture}")
    model = FactoryPartClassifier(
        num_classes=len(class_names),
        architecture=config.model.architecture,
        pretrained=config.model.pretrained,
        dropout_rate=config.model.dropout_rate,
        freeze_backbone=config.model.freeze_backbone,
    )
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"   Parameter: {trainable:,} trainierbar / {total:,} total")
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Early Stopping
    early_stopping = None
    if config.training.early_stopping.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping.get("patience", 5),
            min_delta=config.training.early_stopping.get("min_delta", 0.001),
        )
    
    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    logger = TrainingLogger(run_dir)
    
    # Training Loop
    print(f"\n🚀 Training startet ({config.training.epochs} Epochen)...\n")
    
    best_accuracy = 0.0
    
    for epoch in range(1, config.training.epochs + 1):
        model.train()
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
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        eval_results = evaluate_model(model, val_loader, criterion, device, class_names)
        
        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log(epoch, {
            "train_loss": train_loss,
            "val_loss": eval_results["val_loss"],
            "val_accuracy": eval_results["val_accuracy"],
            "val_f1": eval_results["val_f1"],
            "learning_rate": current_lr,
        })
        
        # Scheduler Step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_results["val_accuracy"])
            else:
                scheduler.step()
        
        # Best Model speichern
        if eval_results["val_accuracy"] > best_accuracy:
            best_accuracy = eval_results["val_accuracy"]
            best_model_path = MODELS_DIR / "factory_cnn_best.pt"
            save_model(model, best_model_path, metadata={
                "accuracy": best_accuracy,
                "epoch": epoch,
                "architecture": config.model.architecture,
            })
        
        # Early Stopping Check
        if early_stopping:
            if early_stopping(eval_results["val_accuracy"], model):
                print(f"\n⏹️ Early Stopping nach Epoch {epoch}")
                model.load_state_dict(early_stopping.best_model_state)
                break
    
    # Finale Evaluation
    print("\n" + "=" * 60)
    print("FINALE EVALUATION")
    print("=" * 60)
    
    final_results = evaluate_model(model, val_loader, criterion, device, class_names)
    
    # Report ausgeben
    report = generate_evaluation_report(final_results["metrics"], class_names)
    print(report)
    
    # Metriken speichern
    final_results["metrics"].save(run_dir / "final_metrics.json")
    
    # Confusion Matrix speichern
    save_confusion_matrix(
        final_results["labels"],
        final_results["probs"],
        class_names,
        PLOTS_DIR / "final_confusion_matrix.png",
    )
    
    # Training Curves speichern
    logger.plot_curves(PLOTS_DIR / "training_curves.png")
    logger.save()
    
    # Finales Modell speichern
    final_model_path = MODELS_DIR / f"factory_cnn_{timestamp}.pt"
    save_model(model, final_model_path, metadata={
        "accuracy": final_results["val_accuracy"],
        "f1": final_results["val_f1"],
        "architecture": config.model.architecture,
        "timestamp": timestamp,
    })
    
    # Auch als "latest" speichern
    save_model(model, MODELS_DIR / "factory_cnn.pt")
    
    print(f"\n✅ Training abgeschlossen!")
    print(f"   Beste Accuracy: {best_accuracy:.2%}")
    print(f"   Modell: {final_model_path}")
    print(f"   Logs: {run_dir}")


if __name__ == "__main__":
    train()
