"""
Professionelle Augmentations-Pipeline für Industrie-Bildklassifikation.
Simuliert realistische Variationen wie sie in Fabriken vorkommen.
"""

from torchvision import transforms
from typing import Dict, Any, Optional
import random
import torch


def get_train_transforms(
    img_size: int = 224,
    config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """
    Starke Augmentations für Training.
    Simuliert: Beleuchtungsvariationen, Kamerawinkel, Verschmutzung.
    """
    config = config or {}
    
    transform_list = [
        transforms.Resize((img_size + 32, img_size + 32)),  # Etwas größer für Crop
        transforms.RandomCrop(img_size),
    ]
    
    # Horizontales Flip (sinnvoll für die meisten Teile)
    if config.get("horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Rotation (simuliert verschiedene Kamerawinkel)
    rotation = config.get("rotation_degrees", 15)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))
    
    # Color Jitter (simuliert Beleuchtungsvariationen in der Fabrik)
    color_config = config.get("color_jitter", {})
    if color_config:
        transform_list.append(transforms.ColorJitter(
            brightness=color_config.get("brightness", 0.2),
            contrast=color_config.get("contrast", 0.2),
            saturation=color_config.get("saturation", 0.2),
            hue=color_config.get("hue", 0.1),
        ))
    
    # Gaussian Blur (simuliert Unschärfe durch Bewegung/Fokus)
    blur_prob = config.get("gaussian_blur", 0.1)
    if blur_prob > 0:
        transform_list.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=blur_prob))
    
    # Perspektivische Transformation (simuliert verschiedene Kamerawinkel)
    transform_list.append(transforms.RandomPerspective(distortion_scale=0.1, p=0.2))
    
    # Zu Tensor konvertieren
    transform_list.append(transforms.ToTensor())
    
    # Random Erasing (simuliert Verdeckungen/Verschmutzungen)
    erasing_prob = config.get("random_erasing", 0.1)
    if erasing_prob > 0:
        transform_list.append(transforms.RandomErasing(
            p=erasing_prob,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
        ))
    
    # ImageNet Normalisierung (wichtig für Pretrained Models)
    transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transform_list)


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Minimale Transforms für Validation/Test - keine Augmentations."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """Transforms für Produktion/Inferenz."""
    return get_val_transforms(img_size)


def get_tta_transforms(img_size: int = 224) -> list:
    """
    Test-Time Augmentation: Mehrere Varianten eines Bildes für robustere Vorhersagen.
    Verwendet für kritische Anwendungen wo Accuracy wichtiger als Latenz ist.
    """
    base_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return [
        # Original
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # Horizontal Flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # Leichte Rotation links
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation((5, 5)),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # Leichte Rotation rechts
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation((-5, -5)),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # Center Crop
        transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            base_normalize,
        ]),
    ]


class FactoryNoiseSimulator:
    """
    Simuliert realistische Störungen aus Fabrikumgebungen.
    Nützlich um Robustheit des Modells zu testen.
    """
    
    def __init__(self, severity: str = "medium"):
        """
        Args:
            severity: "light", "medium", "heavy"
        """
        self.severity = severity
        self.severity_params = {
            "light": {"noise": 0.02, "blur": 1, "brightness": 0.1},
            "medium": {"noise": 0.05, "blur": 2, "brightness": 0.2},
            "heavy": {"noise": 0.1, "blur": 3, "brightness": 0.3},
        }
    
    def add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fügt Sensorrauschen hinzu."""
        params = self.severity_params[self.severity]
        noise = torch.randn_like(tensor) * params["noise"]
        return torch.clamp(tensor + noise, 0, 1)
    
    def add_dust_spots(self, tensor: torch.Tensor, num_spots: int = 5) -> torch.Tensor:
        """Simuliert Staub auf der Kameralinse."""
        result = tensor.clone()
        _, h, w = tensor.shape
        
        for _ in range(num_spots):
            cx, cy = random.randint(0, w-1), random.randint(0, h-1)
            radius = random.randint(2, 8)
            
            for i in range(max(0, cy-radius), min(h, cy+radius)):
                for j in range(max(0, cx-radius), min(w, cx+radius)):
                    if (i-cy)**2 + (j-cx)**2 <= radius**2:
                        result[:, i, j] *= 0.7  # Abdunkeln
        
        return result
    
    def simulate_motion_blur(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simuliert Bewegungsunschärfe (Fließband)."""
        # Vereinfachte Version - in Produktion würde man scipy/cv2 nutzen
        params = self.severity_params[self.severity]
        kernel_size = params["blur"] * 2 + 1
        
        blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=params["blur"])
        return blur(tensor)
