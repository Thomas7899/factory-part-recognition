# 🏭 Factory Part Recognition AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-End Machine Learning System für die automatische Klassifikation von Industriebauteilen**

Ein produktionsnahes ML-Projekt, das CNN-basierte Bildklassifikation mit einem modernen Web-Dashboard kombiniert. Entwickelt als Portfolio-Projekt mit Fokus auf **industrielle Best Practices**.

![Dashboard Preview](docs/assets/dashboard_preview.png)

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Architektur](#-architektur)
- [Schnellstart](#-schnellstart)
- [Projektstruktur](#-projektstruktur)
- [Modell & Training](#-modell--training)
- [API Dokumentation](#-api-dokumentation)
- [Evaluation & Metriken](#-evaluation--metriken)
- [Konfiguration](#%EF%B8%8F-konfiguration)
- [MLOps Features](#-mlops-features)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)

---

## ✨ Features

### 🤖 Machine Learning
- **ResNet18 Transfer Learning** - Pretrained auf ImageNet, fine-tuned auf 50 Industrieteil-Klassen
- **Confidence Scoring** - Threshold-basierte Klassifikation mit Reject-Option
- **OOD Detection** - Erkennung von unbekannten/invaliden Bildern
- **Monte Carlo Dropout** - Unsicherheitsquantifizierung für kritische Anwendungen
- **Professionelle Augmentations** - Simulation von Industriebedingungen

### 🚀 Production Features
- **FastAPI Backend** - Async, hochperformant, auto-generierte OpenAPI Docs
- **Batch Inference** - Verarbeitung mehrerer Bilder in einem Request
- **Health Checks** - Monitoring-ready Endpoints
- **Model Versioning** - Tracking welches Modell Vorhersagen gemacht hat
- **Input Validation** - Dateityp, Größe, Format-Checks

### 📊 Dashboard
- **Drag & Drop Upload** - Intuitive Bildanalyse
- **Echtzeit-Inferenz** - Sofortige Klassifikation
- **Confidence Visualization** - Farbcodierte Sicherheitsindikatoren
- **History & Persistenz** - SQLite-basierte Speicherung

---

## 🏗 Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/JS)                        │
│                    Tailwind CSS + Drag & Drop                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ /upload  │  │ /predict │  │ /health  │  │ /stats   │        │
│  │          │  │  /batch  │  │          │  │          │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │   PyTorch Model    │   │     SQLite DB     │
        │    ResNet18        │   │   Predictions     │
        │   50 Classes       │   │    + Metadata     │
        └───────────────────┘   └───────────────────┘
```

---

## 🚀 Schnellstart

### Voraussetzungen
- Python 3.10+
- pip oder conda

### Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/factory-part-recognition.git
cd factory-part-recognition

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencies installieren
pip install -r requirements.txt
```

### Training (Optional)

```bash
# Mit Standard-Konfiguration
python src/train_improved.py

# Oder mit Custom Config
python src/train_improved.py --config config/config.yaml
```

### Server starten

```bash
uvicorn main:app --reload
```

### Dashboard öffnen

Öffne `index.html` im Browser oder navigiere zu:
- **Dashboard:** `file:///path/to/index.html`
- **API Docs:** http://127.0.0.1:8000/docs
- **Health Check:** http://127.0.0.1:8000/health

---

## 📁 Projektstruktur

```
factory-part-recognition/
├── 📁 config/
│   └── config.yaml          # Zentrale Konfiguration
├── 📁 data/
│   └── car-parts-50/        # Dataset (train/val/test)
├── 📁 docs/
│   ├── evaluation_report.txt
│   └── assets/
├── 📁 logs/                  # Training Logs
├── 📁 models/
│   ├── factory_cnn.pt       # Produktionsmodell
│   └── factory_cnn_best.pt  # Best Checkpoint
├── 📁 plots/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── calibration_curve.png
├── 📁 src/
│   ├── augmentations.py     # Data Augmentation Pipeline
│   ├── config_loader.py     # Config Management
│   ├── evaluate.py          # Test Evaluation
│   ├── metrics.py           # Industrie-Metriken
│   ├── model.py             # Modell-Definitionen
│   ├── ood_detection.py     # Out-of-Distribution Detection
│   ├── train_improved.py    # Training Pipeline
│   └── utils.py             # Hilfsfunktionen
├── 📁 static/               # Uploaded Images
├── index.html               # Dashboard Frontend
├── main.py                  # FastAPI Backend
├── requirements.txt
└── README.md
```

---

## 🧠 Modell & Training

### Architektur

| Komponente | Details |
|------------|---------|
| **Backbone** | ResNet18 (pretrained ImageNet) |
| **Classifier** | Custom Head mit Dropout (0.3) |
| **Input Size** | 224 × 224 × 3 |
| **Output** | 50 Klassen (Softmax) |
| **Parameter** | ~11.2M (trainierbar) |

### Training Features

- ✅ **Transfer Learning** - ImageNet Weights
- ✅ **Data Augmentation** - Rotation, Color Jitter, Random Erasing
- ✅ **Early Stopping** - Patience 5, basierend auf Val Accuracy
- ✅ **LR Scheduling** - Cosine Annealing
- ✅ **Regularization** - Dropout + Weight Decay

### Augmentation Pipeline

```python
# Simuliert realistische Fabrikbedingungen:
- Random Rotation (±15°)       # Verschiedene Kamerawinkel
- Color Jitter                 # Beleuchtungsvariationen
- Gaussian Blur                # Bewegungsunschärfe
- Random Erasing               # Verdeckungen/Verschmutzungen
- Perspective Transform        # Perspektivische Verzerrung
```

---

## 📡 API Dokumentation

### Endpoints

| Method | Endpoint | Beschreibung |
|--------|----------|--------------|
| `GET` | `/health` | System Health Check |
| `GET` | `/classes` | Liste aller Klassen |
| `GET` | `/images` | Gespeicherte Predictions |
| `GET` | `/stats` | Inferenz-Statistiken |
| `POST` | `/upload` | Einzelbild-Klassifikation |
| `POST` | `/predict/batch` | Batch-Inferenz |

### Beispiel Response

```json
{
  "filename": "brake_pad_001.jpg",
  "prediction": "BRAKE PAD",
  "confidence": 0.94,
  "is_confident": true,
  "status": "success",
  "inference_time_ms": 45.2,
  "model_version": "1.0.0",
  "top_3_predictions": [
    {"class": "BRAKE PAD", "confidence": 0.94},
    {"class": "BRAKE ROTOR", "confidence": 0.03},
    {"class": "BRAKE CALIPER", "confidence": 0.02}
  ]
}
```

### Confidence Status

| Status | Confidence | Bedeutung |
|--------|------------|-----------|
| `success` | ≥ 70% | Sichere Klassifikation |
| `low_confidence` | 50-70% | Unsichere Klassifikation |
| `rejected` | < 50% | Als "UNCERTAIN" markiert |

---

## 📊 Evaluation & Metriken

### Über Standard-Accuracy hinaus

| Metrik | Beschreibung | Wert* |
|--------|--------------|-------|
| **Accuracy** | Overall Correct | ~85% |
| **Top-3 Accuracy** | Richtige Klasse in Top-3 | ~95% |
| **Macro F1** | Durchschnitt über Klassen | ~82% |
| **High-Conf Accuracy** | Accuracy wenn conf > 70% | ~92% |
| **Rejection Rate** | Anteil unsicherer Vorhersagen | ~8% |
| **ECE** | Expected Calibration Error | 0.05 |

*Beispielwerte - tatsächliche Werte abhängig vom Training

### Generierte Plots

- **Confusion Matrix** - Identifiziert Verwechslungen zwischen Klassen
- **Training Curves** - Loss, Accuracy, Learning Rate über Epochen
- **Calibration Curve** - Reliability Diagram für Confidence
- **Per-Class Performance** - F1/Precision/Recall pro Klasse

---

## ⚙️ Konfiguration

Die zentrale `config/config.yaml` ermöglicht reproduzierbare Experimente:

```yaml
# Auszug aus config.yaml
training:
  epochs: 15
  batch_size: 32
  learning_rate: 0.001
  early_stopping:
    enabled: true
    patience: 5

inference:
  confidence_threshold: 0.7  # Minimum für "sicher"
  reject_threshold: 0.5      # Unter diesem Wert: UNCERTAIN

augmentation:
  train:
    horizontal_flip: true
    rotation_degrees: 15
    color_jitter:
      brightness: 0.2
```

---

## 🔄 MLOps Features

### Implementiert

- ✅ **Config-basiertes Training** - YAML Konfiguration
- ✅ **Reproduzierbarkeit** - Seed-Setting, deterministische Ops
- ✅ **Model Checkpointing** - Best Model + Periodic Saves
- ✅ **Training Logging** - JSON History + Curves
- ✅ **Model Versioning** - Version in Predictions gespeichert
- ✅ **Health Monitoring** - `/health` Endpoint

### Geplant (Roadmap)

- 🔜 **MLflow Integration** - Experiment Tracking
- 🔜 **DVC** - Data Version Control
- 🔜 **Docker** - Containerisierung
- 🔜 **GitHub Actions** - CI/CD Pipeline

---

## 🛠 Tech Stack

| Kategorie | Technologie |
|-----------|-------------|
| **ML Framework** | PyTorch 2.2, Torchvision |
| **Backend** | FastAPI, Uvicorn |
| **Database** | SQLAlchemy, SQLite |
| **Frontend** | Vanilla JS, TailwindCSS |
| **Data Science** | NumPy, Scikit-learn, Matplotlib |

---

## 📈 Roadmap

### Phase 1: Core ML ✅
- [x] ResNet18 Training Pipeline
- [x] Transfer Learning
- [x] Data Augmentation
- [x] Evaluation Metriken

### Phase 2: Production ✅
- [x] FastAPI Backend
- [x] Confidence Thresholds
- [x] Batch Inference
- [x] Health Checks

### Phase 3: MLOps 🔄
- [x] Config Management
- [x] Reproducible Training
- [ ] Docker Container
- [ ] CI/CD Pipeline
- [ ] Model Registry (MLflow)

### Phase 4: Advanced 🔜
- [ ] Grad-CAM Visualizations
- [ ] A/B Testing Framework
- [ ] Edge Deployment (ONNX)
- [ ] Active Learning Pipeline

---

## 🤝 Beitragen

Contributions sind willkommen! Bitte:

1. Fork das Repository
2. Erstelle einen Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit deine Änderungen (`git commit -m 'Add AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Öffne einen Pull Request

---

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

---

## 📞 Kontakt

**Projekt Link:** [https://github.com/yourusername/factory-part-recognition](https://github.com/yourusername/factory-part-recognition)

---

<p align="center">
  <i>Entwickelt als Portfolio-Projekt für ML Engineering Rollen</i>
</p>
