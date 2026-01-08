# Factory Part Recognition AI

Ein End-to-End Machine Learning Projekt zur Erkennung von Industrieteilen. Es umfasst das Training eines ResNet-Modells mit PyTorch und die Bereitstellung über eine FastAPI-Schnittstelle mit einem modernen Drag & Drop Dashboard.

## 📸 Dashboard
Das Projekt bietet ein Web-Interface, um Teile per Drag & Drop zu analysieren.
- **Backend:** FastAPI & SQLite
- **Frontend:** HTML5, TailwindCSS, SortableJS (Kein Build-Step nötig)
- **Features:** Echtzeit-Inferenz, Speicherung der Ergebnisse, Confidence-Score-Visualisierung.

## 🚀 Features
- **CNN-Klassifikation:** Unterscheidung von 50 Industrie-Teile-Klassen (ResNet18).
- **Model Training:** PyTorch Skripte für Data Loading, Training und Evaluation.
- **REST API:** Upload-Endpunkte und Abruf der Historie.
- **Persistenz:** Speicherung der Analyse-Ergebnisse in einer SQLite-Datenbank.

## 🛠 Tech Stack
- **ML & Data:** Python, PyTorch, Torchvision, PIL, NumPy
- **Backend:** FastAPI, Uvicorn, SQLAlchemy, Python-Multipart
- **Frontend:** Vanilla JS, TailwindCSS

