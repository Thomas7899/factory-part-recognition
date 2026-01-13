"""
Zentrale Konfigurationsverwaltung für das Projekt.
Lädt YAML-Config und stellt sie als Objekt bereit.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class DataConfig:
    root_dir: str = "data/car-parts-50"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    img_size: int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    architecture: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 50
    dropout_rate: float = 0.3
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    early_stopping: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    train: Dict[str, Any] = field(default_factory=dict)
    val: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.7
    reject_threshold: float = 0.5
    batch_size: int = 16
    enable_tta: bool = False


@dataclass
class OODConfig:
    enabled: bool = True
    max_softmax_threshold: float = 0.85
    entropy_threshold: float = 1.5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    seed: int = 42
    deterministic: bool = True


def load_config(config_path: Optional[Path] = None) -> Config:
    """Lädt Konfiguration aus YAML-Datei."""
    path = config_path or DEFAULT_CONFIG_PATH
    
    if not path.exists():
        print(f"⚠️ Config nicht gefunden: {path}, verwende Defaults")
        return Config()
    
    with open(path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    config = Config(
        data=DataConfig(**yaml_config.get("data", {})),
        model=ModelConfig(**yaml_config.get("model", {})),
        training=TrainingConfig(**yaml_config.get("training", {})),
        augmentation=AugmentationConfig(**yaml_config.get("augmentation", {})),
        inference=InferenceConfig(**yaml_config.get("inference", {})),
        ood=OODConfig(**yaml_config.get("ood", {})),
        seed=yaml_config.get("seed", 42),
        deterministic=yaml_config.get("deterministic", True),
    )
    
    return config


def get_device(preference: str = "auto") -> torch.device:
    """Ermittelt das beste verfügbare Gerät."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def set_seed(seed: int, deterministic: bool = True):
    """Setzt Seeds für Reproduzierbarkeit."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Singleton für einfachen Zugriff
_config: Optional[Config] = None

def get_config() -> Config:
    """Gibt die geladene Konfiguration zurück (Singleton)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
