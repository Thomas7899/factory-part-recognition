"""
Vollständige Test-Evaluation auf dem Test-Set.
Generiert umfassenden Bericht für Portfolio.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Lokale Imports
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config, get_device
from augmentations import get_val_transforms
from model import FactoryPartClassifier
from metrics import full_evaluation, generate_evaluation_report, identify_problematic_classes


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"


def load_test_data(config):
    """Lädt Test-Datensatz."""
    data_dir = PROJECT_ROOT / config.data.root_dir / config.data.test_dir
    
    if not data_dir.exists():
        print(f"⚠️ Test-Verzeichnis nicht gefunden: {data_dir}")
        return None, None
    
    transform = get_val_transforms(config.data.img_size)
    test_dataset = ImageFolder(str(data_dir), transform=transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,  # Auf 0 setzen für macOS Kompatibilität
    )
    
    return test_loader, test_dataset.classes


def plot_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
):
    """Erstellt Reliability Diagram für Calibration."""
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_counts.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reliability Diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.bar(
        bin_confidences, bin_accuracies,
        width=1/n_bins * 0.8,
        alpha=0.7,
        edgecolor="black",
        label="Model"
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confidence Distribution
    ax = axes[1]
    ax.hist(confidences, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(0.7, color="red", linestyle="--", label="Threshold (0.7)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Calibration Plot gespeichert: {save_path}")


def plot_per_class_performance(
    metrics,
    class_names: list,
    save_path: Path,
):
    """Visualisiert Per-Class Performance."""
    
    # Daten extrahieren und sortieren nach F1
    classes = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for name in class_names:
        if name in metrics.per_class_metrics:
            classes.append(name)
            f1_scores.append(metrics.per_class_metrics[name]["f1"])
            precisions.append(metrics.per_class_metrics[name]["precision"])
            recalls.append(metrics.per_class_metrics[name]["recall"])
    
    # Sortieren
    sorted_idx = np.argsort(f1_scores)
    classes = [classes[i] for i in sorted_idx]
    f1_scores = [f1_scores[i] for i in sorted_idx]
    precisions = [precisions[i] for i in sorted_idx]
    recalls = [recalls[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    y_pos = np.arange(len(classes))
    bar_height = 0.25
    
    # Bars
    bars1 = ax.barh(y_pos - bar_height, precisions, bar_height, label="Precision", alpha=0.8)
    bars2 = ax.barh(y_pos, recalls, bar_height, label="Recall", alpha=0.8)
    bars3 = ax.barh(y_pos + bar_height, f1_scores, bar_height, label="F1-Score", alpha=0.8)
    
    # Threshold Line
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.5, label="Threshold (0.7)")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("Per-Class Performance (Sorted by F1)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Per-Class Performance gespeichert: {save_path}")


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
):
    """Erstellt normalisierte Confusion Matrix Heatmap."""
    
    # Normalisieren
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    im = ax.imshow(cm_normalized, cmap="Blues")
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Confusion Matrix Heatmap gespeichert: {save_path}")


def run_evaluation(model_path: Path = None):
    """Führt vollständige Evaluation durch."""
    
    config = load_config()
    device = get_device()
    
    # Model Path
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "factory_cnn.pt"
    
    print(f"🧠 Lade Modell: {model_path}")
    
    # Test Data
    test_loader, class_names = load_test_data(config)
    
    if test_loader is None:
        print("❌ Keine Test-Daten verfügbar")
        return
    
    print(f"📂 Test Samples: {len(test_loader.dataset)}")
    
    # Modell laden
    model = FactoryPartClassifier(
        num_classes=len(class_names),
        architecture=config.model.architecture,
        pretrained=False,
    )
    
    # Versuche State Dict zu laden
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Altes Format - kompatibel laden
            from torchvision import models
            old_model = models.resnet18(weights=None)
            old_model.fc = torch.nn.Linear(old_model.fc.in_features, len(class_names))
            old_model.load_state_dict(checkpoint)
            
            # Transfer weights
            model.backbone.load_state_dict(
                {k: v for k, v in old_model.state_dict().items() if not k.startswith("fc.")},
                strict=False
            )
            model.classifier[4].weight.data = old_model.fc.weight.data
            model.classifier[4].bias.data = old_model.fc.bias.data
            
    except Exception as e:
        print(f"⚠️ Fehler beim Laden: {e}")
        print("   Versuche alternatives Laden...")
        
        from torchvision import models
        old_model = models.resnet18(weights=None)
        old_model.fc = torch.nn.Linear(512, len(class_names))
        old_model.load_state_dict(torch.load(model_path, map_location=device))
        old_model = old_model.to(device)
        old_model.eval()
        
        # Nutze altes Modell direkt
        model = old_model
    
    model = model.to(device)
    model.eval()
    
    # Inference
    print("\n🔄 Evaluiere auf Test-Set...")
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Metriken berechnen
    metrics = full_evaluation(
        probs=all_probs,
        labels=all_labels,
        class_names=class_names,
    )
    
    # Report
    report = generate_evaluation_report(metrics, class_names)
    print(report)
    
    # Report speichern
    report_path = PROJECT_ROOT / "docs" / "evaluation_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n📄 Report gespeichert: {report_path}")
    
    # Plots erstellen
    print("\n📊 Erstelle Visualisierungen...")
    
    plot_calibration_curve(
        all_probs, all_labels,
        PLOTS_DIR / "calibration_curve.png"
    )
    
    plot_per_class_performance(
        metrics, class_names,
        PLOTS_DIR / "per_class_performance.png"
    )
    
    plot_confusion_matrix_heatmap(
        metrics.confusion_matrix, class_names,
        PLOTS_DIR / "confusion_matrix_heatmap.png"
    )
    
    # Metriken als JSON speichern
    metrics.save(PROJECT_ROOT / "docs" / "test_metrics.json")
    
    # Problematische Klassen identifizieren
    problematic = identify_problematic_classes(metrics)
    if problematic:
        print("\n⚠️ PROBLEMATISCHE KLASSEN (F1 < 0.7):")
        for name, class_metrics in problematic:
            print(f"   {name}: F1={class_metrics['f1']:.2f}")
    
    print("\n✅ Evaluation abgeschlossen!")


if __name__ == "__main__":
    run_evaluation()
