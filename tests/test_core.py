"""
Unit Tests für das Factory Part Recognition System.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FactoryPartClassifier, count_parameters
from src.metrics import (
    calculate_top_k_accuracy,
    calculate_calibration_error,
    evaluate_with_rejection,
    full_evaluation,
)
from src.ood_detection import OODDetector, OODMethod
from src.augmentations import get_train_transforms, get_val_transforms


class TestModel:
    """Tests für das Modell."""
    
    def test_model_creation(self):
        """Test ob Modell erstellt werden kann."""
        model = FactoryPartClassifier(num_classes=50)
        assert model is not None
        assert model.num_classes == 50
    
    def test_model_forward_pass(self):
        """Test Forward Pass."""
        model = FactoryPartClassifier(num_classes=50)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 50)
    
    def test_model_with_features(self):
        """Test Forward Pass mit Feature-Rückgabe."""
        model = FactoryPartClassifier(num_classes=50)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        logits, features = model(x, return_features=True)
        
        assert logits.shape == (1, 50)
        assert features.shape[0] == 1
    
    def test_parameter_count(self):
        """Test Parameter-Zählung."""
        model = FactoryPartClassifier(num_classes=50)
        trainable, total = count_parameters(model)
        
        assert trainable > 0
        assert total >= trainable
    
    def test_different_architectures(self):
        """Test verschiedene Backbone-Architekturen."""
        for arch in ["resnet18", "resnet34"]:
            model = FactoryPartClassifier(
                num_classes=50,
                architecture=arch,
                pretrained=False,
            )
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            assert output.shape == (1, 50), f"Failed for {arch}"


class TestMetrics:
    """Tests für Metriken."""
    
    def test_top_k_accuracy(self):
        """Test Top-K Accuracy Berechnung."""
        # Simulierte Wahrscheinlichkeiten
        probs = np.array([
            [0.1, 0.6, 0.2, 0.1],  # Klasse 1 ist Top-1
            [0.3, 0.2, 0.4, 0.1],  # Klasse 2 ist Top-1
        ])
        labels = np.array([1, 2])  # Beide richtig
        
        top1 = calculate_top_k_accuracy(probs, labels, k=1)
        top2 = calculate_top_k_accuracy(probs, labels, k=2)
        
        assert top1 == 1.0  # Beide Top-1 richtig
        assert top2 == 1.0
    
    def test_calibration_error(self):
        """Test Calibration Error Berechnung."""
        # Gut kalibriertes Modell
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ])
        labels = np.array([0, 0, 0])
        
        ece, mce, _, _ = calculate_calibration_error(probs, labels, n_bins=5)
        
        assert 0 <= ece <= 1
        assert 0 <= mce <= 1
    
    def test_rejection_evaluation(self):
        """Test Rejection-basierte Evaluation."""
        probs = np.array([
            [0.9, 0.1],   # Confident, richtig
            [0.55, 0.45], # Unsicher, richtig
            [0.4, 0.6],   # Sollte rejected werden
        ])
        labels = np.array([0, 0, 0])
        
        acc, rej_rate, coverage = evaluate_with_rejection(
            probs, labels, confidence_threshold=0.7
        )
        
        assert 0 <= acc <= 1
        assert 0 <= rej_rate <= 1
        assert rej_rate + coverage == pytest.approx(1.0)
    
    def test_full_evaluation(self):
        """Test vollständige Evaluation."""
        np.random.seed(42)
        
        # Simulierte Daten
        probs = np.random.dirichlet(np.ones(5), size=100)
        labels = np.random.randint(0, 5, size=100)
        class_names = ["A", "B", "C", "D", "E"]
        
        metrics = full_evaluation(probs, labels, class_names)
        
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.macro_f1 <= 1
        assert len(metrics.per_class_metrics) == 5


class TestAugmentations:
    """Tests für Augmentations."""
    
    def test_train_transforms(self):
        """Test Training Transforms."""
        transform = get_train_transforms(img_size=224)
        assert transform is not None
    
    def test_val_transforms(self):
        """Test Validation Transforms."""
        transform = get_val_transforms(img_size=224)
        assert transform is not None
    
    def test_transform_output_shape(self):
        """Test dass Transforms korrektes Output-Shape liefern."""
        from PIL import Image
        
        transform = get_val_transforms(img_size=224)
        
        # Dummy Image
        img = Image.new('RGB', (100, 100), color='red')
        tensor = transform(img)
        
        assert tensor.shape == (3, 224, 224)


class TestOODDetection:
    """Tests für OOD Detection."""
    
    @pytest.fixture
    def model(self):
        """Erstellt Test-Modell."""
        return FactoryPartClassifier(num_classes=50)
    
    def test_ood_detector_creation(self, model):
        """Test OOD Detector Erstellung."""
        detector = OODDetector(
            model=model,
            device=torch.device("cpu"),
        )
        assert detector is not None
    
    def test_max_softmax_detection(self, model):
        """Test Max-Softmax OOD Detection."""
        detector = OODDetector(
            model=model,
            device=torch.device("cpu"),
            softmax_threshold=0.5,
        )
        
        x = torch.randn(1, 3, 224, 224)
        result = detector.detect(x, method=OODMethod.MAX_SOFTMAX)
        
        assert hasattr(result, 'is_ood')
        assert hasattr(result, 'confidence')
        assert 0 <= result.confidence <= 1
    
    def test_entropy_detection(self, model):
        """Test Entropy-basierte OOD Detection."""
        detector = OODDetector(
            model=model,
            device=torch.device("cpu"),
        )
        
        x = torch.randn(1, 3, 224, 224)
        result = detector.detect(x, method=OODMethod.ENTROPY)
        
        assert hasattr(result, 'ood_score')
        assert result.method == "entropy"
    
    def test_ensemble_detection(self, model):
        """Test Ensemble OOD Detection."""
        detector = OODDetector(
            model=model,
            device=torch.device("cpu"),
        )
        
        x = torch.randn(1, 3, 224, 224)
        result = detector.detect_ensemble(x)
        
        assert result.method == "ensemble"
        assert "ood_votes" in result.details


class TestIntegration:
    """Integration Tests."""
    
    def test_full_pipeline(self):
        """Test komplette Pipeline von Modell bis Metriken."""
        # Modell erstellen
        model = FactoryPartClassifier(num_classes=10)
        model.eval()
        
        # Dummy Batch
        x = torch.randn(8, 3, 224, 224)
        
        # Forward Pass
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()
        
        # Dummy Labels
        labels = np.random.randint(0, 10, size=8)
        class_names = [f"Class_{i}" for i in range(10)]
        
        # Evaluation
        metrics = full_evaluation(probs, labels, class_names)
        
        assert metrics is not None
        assert metrics.accuracy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
