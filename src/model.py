"""
Modell-Definitionen mit professionellen Features:
- Verschiedene Backbone-Architekturen
- Dropout für Regularisierung
- Confidence Calibration Support
- Feature Extraction für OOD Detection
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class FactoryPartClassifier(nn.Module):
    """
    Flexibler Klassifikator für Industrieteile mit:
    - Austauschbarem Backbone
    - Dropout für Unsicherheitsschätzung
    - Feature-Extraktion für OOD-Detection
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        architecture: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        
        # Backbone laden
        self.backbone, self.feature_dim = self._create_backbone(
            architecture, pretrained, freeze_backbone
        )
        
        # Custom Classifier Head mit Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )
        
        # Feature-Dimension für OOD-Detection speichern
        self._features: Optional[torch.Tensor] = None
    
    def _create_backbone(
        self, 
        architecture: str, 
        pretrained: bool,
        freeze: bool
    ) -> Tuple[nn.Module, int]:
        """Erstellt Backbone basierend auf Architektur."""
        
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Entferne Original-Classifier
            
        elif architecture == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif architecture == "mobilenet_v3":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unbekannte Architektur: {architecture}")
        
        # Backbone einfrieren falls gewünscht
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        
        return backbone, feature_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass.
        
        Args:
            x: Input Tensor [B, C, H, W]
            return_features: Wenn True, gibt auch Features zurück (für OOD)
            
        Returns:
            logits: Klassen-Logits [B, num_classes]
            features: (optional) Feature-Vektor [B, feature_dim]
        """
        features = self.backbone(x)
        self._features = features  # Speichere für spätere Analyse
        
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_features(self) -> Optional[torch.Tensor]:
        """Gibt zuletzt berechnete Features zurück."""
        return self._features
    
    def enable_dropout(self):
        """Aktiviert Dropout auch im eval() Modus für MC-Dropout."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout für Unsicherheitsschätzung.
        
        Args:
            x: Input Tensor
            n_samples: Anzahl der Forward Passes
            
        Returns:
            mean_probs: Mittlere Wahrscheinlichkeiten
            std_probs: Standardabweichung (epistemische Unsicherheit)
            predictions: Finale Vorhersagen
        """
        self.eval()
        self.enable_dropout()
        
        all_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
        
        all_probs = torch.stack(all_probs, dim=0)  # [n_samples, B, C]
        
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        predictions = mean_probs.argmax(dim=1)
        
        return mean_probs, std_probs, predictions


def load_model(
    model_path: Path,
    num_classes: int = 50,
    architecture: str = "resnet18",
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
) -> FactoryPartClassifier:
    """
    Lädt ein trainiertes Modell.
    
    Args:
        model_path: Pfad zur .pt Datei
        num_classes: Anzahl Klassen
        architecture: Modell-Architektur
        device: Zielgerät
        strict: Strenge beim Laden der Weights
        
    Returns:
        Geladenes Modell im eval() Modus
    """
    model = FactoryPartClassifier(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=False,  # Weights werden geladen
    )
    
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle für alte Modelle die nur FC-Layer hatten
    if "fc.weight" in state_dict and "classifier.4.weight" not in state_dict:
        # Altes Format - konvertiere
        print("⚠️ Konvertiere altes Modellformat...")
        old_state = state_dict
        
        # Lade Backbone separat
        backbone_model = models.resnet18(weights=None)
        backbone_state = {k: v for k, v in old_state.items() if not k.startswith("fc.")}
        backbone_model.load_state_dict(backbone_state, strict=False)
        
        # Für Kompatibilität: nur Backbone laden
        model.backbone.load_state_dict(backbone_state, strict=False)
        
        # Neuen Classifier mit alten FC-Weights initialisieren (so gut es geht)
        if "fc.weight" in old_state:
            model.classifier[4].weight.data = old_state["fc.weight"]
            model.classifier[4].bias.data = old_state["fc.bias"]
    else:
        model.load_state_dict(state_dict, strict=strict)
    
    model = model.to(device)
    model.eval()
    
    return model


def save_model(
    model: FactoryPartClassifier,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Speichert Modell mit optionalen Metadaten.
    
    Args:
        model: Zu speicherndes Modell
        save_path: Zielpfad
        metadata: Optionale Metadaten (Accuracy, Config, etc.)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": model.architecture,
        "num_classes": model.num_classes,
        "dropout_rate": model.dropout_rate,
    }
    
    if metadata:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, save_path)
    print(f"✅ Modell gespeichert: {save_path}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Zählt trainierbare und totale Parameter."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
