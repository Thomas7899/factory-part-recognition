"""
Grad-CAM Visualisierung für Interpretierbarkeit.
Zeigt welche Bildregionen für die Klassifikation wichtig sind.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Visualisiert welche Bildregionen das Modell für die Klassifikation
    als wichtig erachtet. Essentiell für Debugging und Vertrauensbildung.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = "layer4"):
        """
        Args:
            model: PyTorch Modell (ResNet)
            target_layer: Layer für Grad-CAM (meist letzte Conv-Layer)
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Hook für Target Layer registrieren
        self._register_hooks(target_layer)
    
    def _register_hooks(self, target_layer: str):
        """Registriert Forward und Backward Hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Finde Target Layer
        for name, module in self.model.named_modules():
            if name == target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generiert Grad-CAM Heatmap.
        
        Args:
            input_tensor: Preprocessed Image Tensor [1, C, H, W]
            target_class: Klasse für die CAM generiert wird (None = predicted)
            
        Returns:
            cam: Heatmap als numpy array
            predicted_class: Vorhergesagte Klasse
            confidence: Konfidenz-Score
        """
        # Forward Pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward Pass für Target Class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Grad-CAM berechnen
        # Globaler Durchschnitt der Gradienten
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Gewichtete Summe der Aktivierungen
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU und Normalisierung
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Zu numpy konvertieren und normalisieren
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, confidence
    
    def visualize(
        self,
        image: Image.Image,
        cam: np.ndarray,
        predicted_class: str,
        confidence: float,
        save_path: Optional[Path] = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Erstellt Overlay-Visualisierung.
        
        Args:
            image: Original PIL Image
            cam: Grad-CAM Heatmap
            predicted_class: Klassenname
            confidence: Konfidenz-Score
            save_path: Optional Speicherpfad
            alpha: Transparenz des Overlays
            
        Returns:
            Overlay als numpy array
        """
        # Image zu Array
        img_array = np.array(image.resize((224, 224)))
        
        # CAM zu Colormap
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay erstellen
        overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_array)
        
        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(img_array)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title(f'{predicted_class} ({confidence:.1%})')
            axes[2].axis('off')
            
            plt.suptitle('Grad-CAM Visualization', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return overlay


def analyze_predictions_with_gradcam(
    model: torch.nn.Module,
    image_paths: List[Path],
    class_names: List[str],
    device: torch.device,
    output_dir: Path,
    transform=None,
):
    """
    Analysiert mehrere Bilder mit Grad-CAM.
    
    Args:
        model: Trainiertes Modell
        image_paths: Liste von Bildpfaden
        class_names: Liste der Klassennamen
        device: Torch Device
        output_dir: Ausgabeverzeichnis
        transform: Image Transform
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    output_dir.mkdir(exist_ok=True)
    gradcam = GradCAM(model, target_layer="backbone.layer4")
    
    print(f"\n🔍 Analysiere {len(image_paths)} Bilder mit Grad-CAM...\n")
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            cam, pred_idx, conf = gradcam.generate(input_tensor)
            pred_class = class_names[pred_idx]
            
            save_path = output_dir / f"gradcam_{img_path.stem}.png"
            gradcam.visualize(image, cam, pred_class, conf, save_path)
            
            print(f"  ✅ {img_path.name} -> {pred_class} ({conf:.1%})")
            
        except Exception as e:
            print(f"  ❌ {img_path.name}: {e}")
    
    print(f"\n✅ Grad-CAM Analysen gespeichert in: {output_dir}")


def generate_attention_summary(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    num_samples: int = 3,
) -> dict:
    """
    Generiert Zusammenfassung der Modell-Attention für jede Klasse.
    
    Returns:
        Dictionary mit durchschnittlichen Attention-Maps pro Klasse
    """
    gradcam = GradCAM(model, target_layer="backbone.layer4")
    
    class_cams = {name: [] for name in class_names}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            for i in range(min(len(images), num_samples)):
                img_tensor = images[i:i+1]
                label = labels[i].item()
                
                cam, _, _ = gradcam.generate(img_tensor, target_class=label)
                class_cams[class_names[label]].append(cam)
    
    # Durchschnittliche CAMs berechnen
    avg_cams = {}
    for name, cams in class_cams.items():
        if cams:
            avg_cams[name] = np.mean(cams, axis=0)
    
    return avg_cams


if __name__ == "__main__":
    print("Grad-CAM Modul - Import und Verwendung in anderen Skripten")
