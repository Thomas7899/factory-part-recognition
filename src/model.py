import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

class FactoryPartClassifier(nn.Module):
    def __init__(self, num_classes: int, architecture: str = "resnet18", pretrained: bool = False, dropout_rate: float = 0.3):
        super().__init__()
        
        if architecture == "resnet18":
            backbone = models.resnet18(weights=None)
            num_features = backbone.fc.in_features
        elif architecture == "resnet34":
            backbone = models.resnet34(weights=None)
            num_features = backbone.fc.in_features
        elif architecture == "resnet50":
            backbone = models.resnet50(weights=None)
            num_features = backbone.fc.in_features
        else:
            backbone = models.resnet18(weights=None)
            num_features = backbone.fc.in_features

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def load_trained_model(path: str, device: torch.device, num_classes: int = 50):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    
    arch = "resnet18"
    state_dict = checkpoint
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        arch = checkpoint.get("architecture", "resnet18")
    
    model = FactoryPartClassifier(num_classes=num_classes, architecture=arch)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    
    return model