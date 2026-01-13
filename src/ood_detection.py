"""
Out-of-Distribution (OOD) Detection für Industrieanwendungen.

Erkennt Bilder die nicht zu den trainierten Klassen gehören:
- Unbekannte Teile
- Defekte Bilder
- Falsche Kameraposition
- Verschmutzte Linse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class OODMethod(Enum):
    """Verfügbare OOD-Detection Methoden."""
    MAX_SOFTMAX = "max_softmax"
    ENTROPY = "entropy"
    ENERGY = "energy"
    MAHALANOBIS = "mahalanobis"


@dataclass
class OODResult:
    """Ergebnis der OOD-Detection."""
    is_ood: bool
    confidence: float
    ood_score: float
    method: str
    details: Dict[str, float]


class OODDetector:
    """
    Out-of-Distribution Detector für Produktionsumgebungen.
    
    Verwendet mehrere Methoden um robuste OOD-Erkennung zu ermöglichen.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        softmax_threshold: float = 0.85,
        entropy_threshold: float = 1.5,
        energy_threshold: float = -5.0,
    ):
        """
        Args:
            model: Trainiertes Klassifikationsmodell
            device: Torch Device
            softmax_threshold: Threshold für Max-Softmax Methode
            entropy_threshold: Threshold für Entropy Methode
            energy_threshold: Threshold für Energy Methode
        """
        self.model = model
        self.device = device
        self.thresholds = {
            OODMethod.MAX_SOFTMAX: softmax_threshold,
            OODMethod.ENTROPY: entropy_threshold,
            OODMethod.ENERGY: energy_threshold,
        }
        
        # Feature-Statistiken für Mahalanobis (müssen auf Training-Daten berechnet werden)
        self.class_means: Optional[torch.Tensor] = None
        self.precision_matrix: Optional[torch.Tensor] = None
    
    def detect(
        self,
        x: torch.Tensor,
        method: OODMethod = OODMethod.MAX_SOFTMAX,
    ) -> OODResult:
        """
        Führt OOD-Detection durch.
        
        Args:
            x: Input Tensor [1, C, H, W]
            method: OOD-Detection Methode
            
        Returns:
            OODResult mit Ergebnis und Details
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            probs = F.softmax(logits, dim=1)
        
        if method == OODMethod.MAX_SOFTMAX:
            return self._max_softmax_detection(probs)
        elif method == OODMethod.ENTROPY:
            return self._entropy_detection(probs)
        elif method == OODMethod.ENERGY:
            return self._energy_detection(logits)
        else:
            raise ValueError(f"Unbekannte Methode: {method}")
    
    def detect_ensemble(self, x: torch.Tensor) -> OODResult:
        """
        Kombiniert mehrere OOD-Methoden für robustere Erkennung.
        
        Ein Sample wird als OOD markiert wenn mindestens 2 von 3 Methoden
        es als OOD klassifizieren.
        """
        results = {
            OODMethod.MAX_SOFTMAX: self.detect(x, OODMethod.MAX_SOFTMAX),
            OODMethod.ENTROPY: self.detect(x, OODMethod.ENTROPY),
            OODMethod.ENERGY: self.detect(x, OODMethod.ENERGY),
        }
        
        ood_votes = sum(1 for r in results.values() if r.is_ood)
        is_ood = ood_votes >= 2
        
        # Durchschnittliche Confidence
        avg_confidence = np.mean([r.confidence for r in results.values()])
        
        details = {
            f"{method.value}_ood": result.is_ood
            for method, result in results.items()
        }
        details["ood_votes"] = ood_votes
        
        return OODResult(
            is_ood=is_ood,
            confidence=1.0 - avg_confidence if is_ood else avg_confidence,
            ood_score=ood_votes / len(results),
            method="ensemble",
            details=details,
        )
    
    def _max_softmax_detection(self, probs: torch.Tensor) -> OODResult:
        """
        Maximum Softmax Probability Methode.
        OOD wenn max(softmax) < threshold.
        """
        max_prob = probs.max().item()
        threshold = self.thresholds[OODMethod.MAX_SOFTMAX]
        is_ood = max_prob < threshold
        
        return OODResult(
            is_ood=is_ood,
            confidence=max_prob,
            ood_score=1.0 - max_prob,
            method="max_softmax",
            details={"max_probability": max_prob, "threshold": threshold},
        )
    
    def _entropy_detection(self, probs: torch.Tensor) -> OODResult:
        """
        Entropy-basierte Methode.
        OOD wenn Entropy > threshold (hohe Unsicherheit).
        """
        # Entropy berechnen: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).item()
        threshold = self.thresholds[OODMethod.ENTROPY]
        
        # Normalisierte Entropy (0-1)
        max_entropy = np.log(probs.shape[1])  # log(num_classes)
        normalized_entropy = entropy / max_entropy
        
        is_ood = entropy > threshold
        
        return OODResult(
            is_ood=is_ood,
            confidence=1.0 - normalized_entropy,
            ood_score=normalized_entropy,
            method="entropy",
            details={
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "threshold": threshold,
            },
        )
    
    def _energy_detection(self, logits: torch.Tensor) -> OODResult:
        """
        Energy-basierte OOD Detection.
        E(x) = -T * log(sum(exp(logits/T)))
        OOD wenn Energy < threshold (niedrige Energy = unsicher).
        """
        temperature = 1.0
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1).item()
        threshold = self.thresholds[OODMethod.ENERGY]
        
        is_ood = energy < threshold
        
        # Normalisiere Energy Score für bessere Interpretierbarkeit
        # Typische Range ist etwa -15 bis 0
        normalized_score = max(0, min(1, (energy + 15) / 15))
        
        return OODResult(
            is_ood=is_ood,
            confidence=normalized_score,
            ood_score=1.0 - normalized_score,
            method="energy",
            details={"energy": energy, "threshold": threshold},
        )
    
    def fit_mahalanobis(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_classes: int,
    ):
        """
        Berechnet Klassenstatistiken für Mahalanobis-Distance.
        Sollte auf dem Trainings-Datensatz aufgerufen werden.
        """
        self.model.eval()
        
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                _, features = self.model(images, return_features=True)
                features_list.append(features.cpu())
                labels_list.append(labels)
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Klassenmittelwerte berechnen
        self.class_means = torch.zeros(num_classes, features.shape[1])
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.class_means[c] = features[mask].mean(dim=0)
        
        # Kovarianzmatrix (shared across classes)
        centered = features - self.class_means[labels]
        cov = torch.mm(centered.t(), centered) / features.shape[0]
        
        # Precision Matrix (Inverse der Kovarianz)
        self.precision_matrix = torch.linalg.pinv(cov + 1e-6 * torch.eye(cov.shape[0]))
        
        print(f"✅ Mahalanobis-Statistiken berechnet auf {len(features)} Samples")


def get_rejection_message(ood_result: OODResult) -> str:
    """Generiert eine benutzerfreundliche Nachricht für OOD-Samples."""
    
    if not ood_result.is_ood:
        return ""
    
    confidence_pct = (1 - ood_result.confidence) * 100
    
    messages = {
        "max_softmax": f"⚠️ Niedrige Konfidenz ({confidence_pct:.0f}%). Das Bild könnte ein unbekanntes Teil zeigen.",
        "entropy": f"⚠️ Hohe Unsicherheit. Das Modell kann das Teil nicht eindeutig zuordnen.",
        "energy": f"⚠️ Ungewöhnliches Muster erkannt. Bitte überprüfen Sie das Bild manuell.",
        "ensemble": f"⚠️ Mehrere Indikatoren deuten auf ein unbekanntes Teil hin.",
    }
    
    return messages.get(ood_result.method, "⚠️ Unsichere Klassifikation")


class ProductionOODHandler:
    """
    Handler für OOD in Produktionsumgebungen.
    
    Definiert was passiert wenn ein OOD-Sample erkannt wird:
    - Logging
    - Alert an Operator
    - Speichern für Review
    """
    
    def __init__(
        self,
        ood_detector: OODDetector,
        alert_callback: Optional[callable] = None,
        log_path: Optional[str] = None,
    ):
        self.detector = ood_detector
        self.alert_callback = alert_callback
        self.log_path = log_path
        self.ood_count = 0
        self.total_count = 0
    
    def handle(
        self,
        image_tensor: torch.Tensor,
        image_path: str,
    ) -> Tuple[bool, OODResult]:
        """
        Verarbeitet ein Bild und handelt bei OOD.
        
        Returns:
            (should_classify, ood_result): Ob klassifiziert werden soll
        """
        self.total_count += 1
        
        result = self.detector.detect_ensemble(image_tensor)
        
        if result.is_ood:
            self.ood_count += 1
            self._handle_ood_sample(result, image_path)
            return False, result
        
        return True, result
    
    def _handle_ood_sample(self, result: OODResult, image_path: str):
        """Interne Verarbeitung von OOD-Samples."""
        
        # Alert senden falls konfiguriert
        if self.alert_callback:
            self.alert_callback(f"OOD detected: {image_path}", result)
        
        # Logging
        print(f"🚨 OOD erkannt: {image_path}")
        print(f"   Score: {result.ood_score:.2f}, Details: {result.details}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Gibt OOD-Statistiken zurück."""
        return {
            "total_processed": self.total_count,
            "ood_detected": self.ood_count,
            "ood_rate": self.ood_count / max(1, self.total_count),
        }
