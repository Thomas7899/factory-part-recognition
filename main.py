"""
Factory Part Recognition API
Produktionsreife FastAPI-Implementierung mit:
- Confidence Thresholds & Reject Option
- OOD Detection
- Input Validation
- Batch Inference Support
- Model Versioning
- Health Checks
"""

import os
import sys
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("factory-ai")

# --- Konfiguration ---
class InferenceConfig:
    CONFIDENCE_THRESHOLD = 0.7   # Minimum für "sichere" Vorhersage
    REJECT_THRESHOLD = 0.5       # Unter diesem Wert: "UNCERTAIN"
    MAX_FILE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    MODEL_VERSION = "1.0.0"

# --- Datenbank Setup (SQLAlchemy) ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    prediction = Column(String)
    score = Column(Float)
    is_confident = Column(Boolean, default=True)  # Ob Vorhersage sicher ist
    inference_time_ms = Column(Float, nullable=True)  # Latenz-Tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String, default=InferenceConfig.MODEL_VERSION)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Response Models (Pydantic) ---
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    is_confident: bool
    status: str  # "success", "low_confidence", "rejected", "error"
    inference_time_ms: float
    model_version: str
    top_3_predictions: Optional[List[dict]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str
    num_classes: int
    uptime_seconds: float

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_time_ms: float
    successful: int
    failed: int

# --- App Setup ---
START_TIME = datetime.utcnow()
app = FastAPI(
    title="Factory Part Recognition API",
    description="CNN-basierte Klassifikation von Industrieteilen mit Confidence Scoring",
    version=InferenceConfig.MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- KI Setup ---
TRAIN_DIR = "data/car-parts-50/train"
CLASSES = [
    'AIR COMPRESSOR', 'ALTERNATOR', 'BATTERY', 'BRAKE CALIPER', 'BRAKE PAD',
    'BRAKE ROTOR', 'CAMSHAFT', 'CARBERATOR', 'CLUTCH PLATE', 'COIL SPRING',
    'CRANKSHAFT', 'CYLINDER HEAD', 'DISTRIBUTOR', 'ENGINE BLOCK', 'ENGINE VALVE',
    'FUEL INJECTOR', 'FUSE BOX', 'GAS CAP', 'HEADLIGHTS', 'IDLER ARM',
    'IGNITION COIL', 'INSTRUMENT CLUSTER', 'LEAF SPRING', 'LOWER CONTROL ARM',
    'MUFFLER', 'OIL FILTER', 'OIL PAN', 'OIL PRESSURE SENSOR', 'OVERFLOW TANK',
    'OXYGEN SENSOR', 'PISTON', 'PRESSURE PLATE', 'RADIATOR', 'RADIATOR FAN',
    'RADIATOR HOSE', 'RADIO', 'RIM', 'SHIFT KNOB', 'SIDE MIRROR', 'SPARK PLUG',
    'SPOILER', 'STARTER', 'TAILLIGHTS', 'THERMOSTAT', 'TORQUE CONVERTER',
    'TRANSMISSION', 'VACUUM BRAKE BOOSTER', 'VALVE LIFTER', 'WATER PUMP',
    'WINDOW REGULATOR'
]

if os.path.exists(TRAIN_DIR):
    try:
        class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    except:
        class_names = CLASSES
else:
    class_names = CLASSES

MODEL_PATH = "models/factory_cnn.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        logger.info(f"✅ Modell geladen: {MODEL_PATH}")
        logger.info(f"   Device: {device}, Klassen: {len(class_names)}")
        return model
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden: {e}")
        return None

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def validate_image_file(filename: str, file_size: int) -> tuple[bool, str]:
    """Validiert hochgeladene Bilder."""
    ext = Path(filename).suffix.lower()
    
    if ext not in InferenceConfig.ALLOWED_EXTENSIONS:
        return False, f"Ungültiges Format: {ext}. Erlaubt: {InferenceConfig.ALLOWED_EXTENSIONS}"
    
    max_size = InferenceConfig.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        return False, f"Datei zu groß: {file_size/1024/1024:.1f}MB. Max: {InferenceConfig.MAX_FILE_SIZE_MB}MB"
    
    return True, ""


def predict_image_advanced(image_path: str) -> dict:
    """
    Erweiterte Vorhersage mit:
    - Top-3 Predictions
    - Confidence Classification
    - Timing
    """
    if model is None:
        return {
            "prediction": "MODEL_ERROR",
            "confidence": 0.0,
            "status": "error",
            "top_3": [],
            "inference_time_ms": 0,
        }

    try:
        start_time = time.time()
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Top-3 Predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            top3_probs = top3_probs.squeeze().cpu().numpy()
            top3_indices = top3_indices.squeeze().cpu().numpy()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Top Prediction
        top_idx = top3_indices[0]
        top_prob = float(top3_probs[0])
        prediction = class_names[top_idx]
        
        # Status basierend auf Confidence
        if top_prob >= InferenceConfig.CONFIDENCE_THRESHOLD:
            status = "success"
            is_confident = True
        elif top_prob >= InferenceConfig.REJECT_THRESHOLD:
            status = "low_confidence"
            is_confident = False
        else:
            status = "rejected"
            prediction = "UNCERTAIN"
            is_confident = False
        
        # Top-3 formatieren
        top_3 = [
            {"class": class_names[idx], "confidence": float(prob)}
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        
        return {
            "prediction": prediction,
            "confidence": top_prob,
            "status": status,
            "is_confident": is_confident,
            "top_3": top_3,
            "inference_time_ms": inference_time,
        }
        
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {
            "prediction": "ERROR",
            "confidence": 0.0,
            "status": "error",
            "is_confident": False,
            "top_3": [],
            "inference_time_ms": 0,
        }


# Legacy Funktion für Kompatibilität
def predict_image(image_path):
    result = predict_image_advanced(image_path)
    return result["prediction"], result["confidence"]


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health Check Endpoint für Monitoring."""
    uptime = (datetime.utcnow() - START_TIME).total_seconds()
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=InferenceConfig.MODEL_VERSION,
        device=str(device),
        num_classes=len(class_names),
        uptime_seconds=uptime,
    )


@app.get("/classes")
def get_classes():
    """Gibt alle verfügbaren Klassen zurück."""
    return {
        "num_classes": len(class_names),
        "classes": class_names,
    }


@app.get("/images")
def get_images(
    db: Session = Depends(get_db),
    confident_only: bool = Query(False, description="Nur sichere Vorhersagen")
):
    """Lädt alle gespeicherten Bilder mit Vorhersagen."""
    query = db.query(ImageRecord)
    
    if confident_only:
        query = query.filter(ImageRecord.is_confident == True)
    
    records = query.all()
    results = []
    
    if os.path.exists("static"):
        files = os.listdir("static")
        image_files = set(f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        
        for record in records:
            if record.filename in image_files:
                results.append({
                    "url": f"http://127.0.0.1:8000/static/{record.filename}",
                    "name": record.filename,
                    "prediction": record.prediction,
                    "score": record.score,
                    "is_confident": record.is_confident,
                    "model_version": record.model_version,
                })
                
    return results[::-1]


@app.post("/upload", response_model=PredictionResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    return_top_3: bool = Query(True, description="Top-3 Predictions zurückgeben")
):
    """
    Lädt ein Bild hoch und führt Klassifikation durch.
    
    - **file**: Bilddatei (JPG, PNG, WebP)
    - **return_top_3**: Optional Top-3 Predictions
    
    Returns:
        Prediction mit Confidence Score und Status
    """
    # Validierung
    content = await file.read()
    await file.seek(0)  # Reset für späteres Lesen
    
    is_valid, error_msg = validate_image_file(file.filename, len(content))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    file_location = f"static/{file.filename}"
    
    # Datei speichern
    with open(file_location, "wb") as buffer:
        buffer.write(content)
    
    # KI Vorhersage
    result = predict_image_advanced(file_location)
    
    prediction_text = f"{result['prediction']} ({result['confidence']:.1%})"
    
    # Datenbank Update oder Insert
    existing_record = db.query(ImageRecord).filter(ImageRecord.filename == file.filename).first()
    
    if existing_record:
        existing_record.prediction = prediction_text
        existing_record.score = result["confidence"]
        existing_record.is_confident = result["is_confident"]
        existing_record.inference_time_ms = result["inference_time_ms"]
        existing_record.model_version = InferenceConfig.MODEL_VERSION
    else:
        new_record = ImageRecord(
            filename=file.filename,
            prediction=prediction_text,
            score=result["confidence"],
            is_confident=result["is_confident"],
            inference_time_ms=result["inference_time_ms"],
            model_version=InferenceConfig.MODEL_VERSION,
        )
        db.add(new_record)
    
    db.commit()
    
    logger.info(f"Prediction: {file.filename} -> {result['prediction']} ({result['confidence']:.1%})")
    
    return PredictionResponse(
        filename=file.filename,
        prediction=result["prediction"],
        confidence=result["confidence"],
        is_confident=result["is_confident"],
        status=result["status"],
        inference_time_ms=result["inference_time_ms"],
        model_version=InferenceConfig.MODEL_VERSION,
        top_3_predictions=result["top_3"] if return_top_3 else None,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    Batch-Inferenz für mehrere Bilder.
    
    Effizienter als einzelne Uploads für größere Mengen.
    """
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            content = await file.read()
            
            is_valid, error_msg = validate_image_file(file.filename, len(content))
            if not is_valid:
                results.append(PredictionResponse(
                    filename=file.filename,
                    prediction="INVALID_FILE",
                    confidence=0.0,
                    is_confident=False,
                    status="error",
                    inference_time_ms=0,
                    model_version=InferenceConfig.MODEL_VERSION,
                ))
                failed += 1
                continue
            
            file_location = f"static/{file.filename}"
            with open(file_location, "wb") as buffer:
                buffer.write(content)
            
            result = predict_image_advanced(file_location)
            
            results.append(PredictionResponse(
                filename=file.filename,
                prediction=result["prediction"],
                confidence=result["confidence"],
                is_confident=result["is_confident"],
                status=result["status"],
                inference_time_ms=result["inference_time_ms"],
                model_version=InferenceConfig.MODEL_VERSION,
                top_3_predictions=result["top_3"],
            ))
            
            if result["status"] != "error":
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Batch error for {file.filename}: {e}")
            failed += 1
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        results=results,
        total_time_ms=total_time,
        successful=successful,
        failed=failed,
    )


@app.get("/stats")
def get_statistics(db: Session = Depends(get_db)):
    """Gibt Inferenz-Statistiken zurück."""
    total = db.query(ImageRecord).count()
    confident = db.query(ImageRecord).filter(ImageRecord.is_confident == True).count()
    uncertain = total - confident
    
    avg_inference_time = db.query(ImageRecord).filter(
        ImageRecord.inference_time_ms != None
    ).with_entities(
        ImageRecord.inference_time_ms
    ).all()
    
    avg_time = np.mean([r[0] for r in avg_inference_time]) if avg_inference_time else 0
    
    return {
        "total_predictions": total,
        "confident_predictions": confident,
        "uncertain_predictions": uncertain,
        "confidence_rate": confident / max(1, total),
        "avg_inference_time_ms": avg_time,
        "model_version": InferenceConfig.MODEL_VERSION,
    }