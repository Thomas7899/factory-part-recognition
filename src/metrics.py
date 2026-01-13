"""
Industrie-relevante Metriken für ML-Evaluierung.
Geht über einfache Accuracy hinaus.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class EvaluationMetrics:
    """Container für alle Evaluierungsmetriken."""
    accuracy: float
    top_k_accuracy: float  # Top-3 oder Top-5
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    
    # Industrie-spezifisch
    rejection_rate: float  # Anteil als "unsicher" markierter Samples
    high_confidence_accuracy: float  # Accuracy nur für sichere Vorhersagen
    
    # Calibration
    expected_calibration_error: float
    max_calibration_error: float
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary für JSON-Export."""
        return {
            "accuracy": self.accuracy,
            "top_k_accuracy": self.top_k_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "rejection_rate": self.rejection_rate,
            "high_confidence_accuracy": self.high_confidence_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "max_calibration_error": self.max_calibration_error,
            "per_class_metrics": self.per_class_metrics,
        }
    
    def save(self, path: Path):
        """Speichert Metriken als JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def calculate_top_k_accuracy(
    probs: np.ndarray, 
    labels: np.ndarray, 
    k: int = 3
) -> float:
    """
    Berechnet Top-K Accuracy.
    Wichtig für Industrie: "War die richtige Klasse unter den Top-3?"
    """
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(labels[i] in top_k_preds[i] for i in range(len(labels)))
    return correct / len(labels)


def calculate_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Berechnet Expected und Maximum Calibration Error.
    
    Ein gut kalibriertes Modell sollte bei 90% Confidence 
    auch 90% der Zeit richtig liegen.
    
    Returns:
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        bin_accuracies: Accuracy pro Bin
        bin_confidences: Mittlere Confidence pro Bin
    """
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
            bin_confidences.append(0)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # ECE: Gewichteter Durchschnitt der |acc - conf| pro Bin
    total = bin_counts.sum()
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total if total > 0 else 0
    
    # MCE: Maximum Calibration Error
    mce = np.max(np.abs(bin_accuracies - bin_confidences))
    
    return ece, mce, bin_accuracies, bin_confidences


def evaluate_with_rejection(
    probs: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float = 0.7,
) -> Tuple[float, float, float]:
    """
    Evaluiert mit Reject-Option.
    
    In der Industrie: Lieber "unsicher" sagen als falsch klassifizieren.
    
    Returns:
        accepted_accuracy: Accuracy für akzeptierte Samples
        rejection_rate: Anteil der abgelehnten Samples
        coverage: Anteil der akzeptierten Samples
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    accepted_mask = confidences >= confidence_threshold
    
    if accepted_mask.sum() == 0:
        return 0.0, 1.0, 0.0
    
    accepted_accuracy = (predictions[accepted_mask] == labels[accepted_mask]).mean()
    rejection_rate = (~accepted_mask).mean()
    coverage = accepted_mask.mean()
    
    return accepted_accuracy, rejection_rate, coverage


def compute_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Berechnet Precision, Recall, F1 pro Klasse.
    Wichtig um schwache Klassen zu identifizieren.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    
    return per_class


def full_evaluation(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    confidence_threshold: float = 0.7,
    top_k: int = 3,
) -> EvaluationMetrics:
    """
    Führt vollständige Evaluierung durch.
    
    Args:
        probs: Wahrscheinlichkeiten [N, C]
        labels: Ground Truth Labels [N]
        class_names: Liste der Klassennamen
        confidence_threshold: Schwelle für Reject-Option
        top_k: K für Top-K Accuracy
        
    Returns:
        EvaluationMetrics mit allen berechneten Metriken
    """
    predictions = np.argmax(probs, axis=1)
    
    # Basis-Metriken
    accuracy = (predictions == labels).mean()
    top_k_acc = calculate_top_k_accuracy(probs, labels, k=top_k)
    
    # Macro/Weighted Metriken
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    # Per-Class
    per_class = compute_per_class_metrics(labels, predictions, class_names)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    
    # Rejection Metrics
    high_conf_acc, rejection_rate, _ = evaluate_with_rejection(
        probs, labels, confidence_threshold
    )
    
    # Calibration
    ece, mce, _, _ = calculate_calibration_error(probs, labels)
    
    return EvaluationMetrics(
        accuracy=accuracy,
        top_k_accuracy=top_k_acc,
        macro_precision=precision,
        macro_recall=recall,
        macro_f1=f1,
        weighted_f1=weighted_f1,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        rejection_rate=rejection_rate,
        high_confidence_accuracy=high_conf_acc,
        expected_calibration_error=ece,
        max_calibration_error=mce,
    )


def identify_problematic_classes(
    metrics: EvaluationMetrics,
    f1_threshold: float = 0.7,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Identifiziert Klassen mit schlechter Performance.
    
    Returns:
        Liste von (Klassenname, Metriken) sortiert nach F1
    """
    problematic = []
    
    for name, class_metrics in metrics.per_class_metrics.items():
        if class_metrics["f1"] < f1_threshold:
            problematic.append((name, class_metrics))
    
    # Sortiere nach F1 (schlechteste zuerst)
    problematic.sort(key=lambda x: x[1]["f1"])
    
    return problematic


def generate_evaluation_report(
    metrics: EvaluationMetrics,
    class_names: List[str],
) -> str:
    """Generiert einen lesbaren Evaluierungsbericht."""
    
    report = []
    report.append("=" * 60)
    report.append("EVALUATION REPORT - Factory Part Recognition")
    report.append("=" * 60)
    
    report.append("\n📊 OVERALL METRICS")
    report.append("-" * 40)
    report.append(f"  Accuracy:              {metrics.accuracy:.2%}")
    report.append(f"  Top-3 Accuracy:        {metrics.top_k_accuracy:.2%}")
    report.append(f"  Macro F1:              {metrics.macro_f1:.2%}")
    report.append(f"  Weighted F1:           {metrics.weighted_f1:.2%}")
    
    report.append("\n🎯 CONFIDENCE METRICS")
    report.append("-" * 40)
    report.append(f"  High-Conf Accuracy:    {metrics.high_confidence_accuracy:.2%}")
    report.append(f"  Rejection Rate:        {metrics.rejection_rate:.2%}")
    
    report.append("\n📈 CALIBRATION")
    report.append("-" * 40)
    report.append(f"  Expected Cal. Error:   {metrics.expected_calibration_error:.4f}")
    report.append(f"  Max Cal. Error:        {metrics.max_calibration_error:.4f}")
    
    # Problematische Klassen
    problematic = identify_problematic_classes(metrics)
    if problematic:
        report.append("\n⚠️ PROBLEMATIC CLASSES (F1 < 0.7)")
        report.append("-" * 40)
        for name, class_metrics in problematic[:10]:  # Top 10
            report.append(
                f"  {name:25} | P: {class_metrics['precision']:.2f} | "
                f"R: {class_metrics['recall']:.2f} | F1: {class_metrics['f1']:.2f}"
            )
    
    # Beste Klassen
    best = sorted(
        metrics.per_class_metrics.items(),
        key=lambda x: x[1]["f1"],
        reverse=True
    )[:5]
    
    report.append("\n✅ BEST PERFORMING CLASSES")
    report.append("-" * 40)
    for name, class_metrics in best:
        report.append(
            f"  {name:25} | P: {class_metrics['precision']:.2f} | "
            f"R: {class_metrics['recall']:.2f} | F1: {class_metrics['f1']:.2f}"
        )
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)
