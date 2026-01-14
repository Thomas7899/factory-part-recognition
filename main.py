import os
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Neue Imports
from src.model import load_trained_model
from src.augmentations import get_val_transforms
from src.config_loader import get_device

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

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Klassen automatisch laden
TRAIN_DIR = "data/car-parts-50/train"
FALLBACK_CLASSES = [f"Class_{i}" for i in range(50)]
class_names = FALLBACK_CLASSES

if os.path.exists(TRAIN_DIR):
    found = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
    if found:
        class_names = found

# Setup Device & Model
device = get_device()
MODEL_PATH = "models/factory_cnn.pt"
model = None

try:
    print(f"Lade Modell auf {device}...")
    model = load_trained_model(MODEL_PATH, device, len(class_names))
    print("✅ Modell erfolgreich geladen (Neues Format).")
except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")

# Transformation
transform = get_val_transforms(224)

def predict_image(image_path):
    if model is None:
        return "Modell Fehler", 0.0

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)

        return class_names[predicted_idx.item()], score.item()
    except Exception as e:
        print(f"Fehler bei Prediction: {e}")
        return "Fehler", 0.0

@app.get("/images")
def get_images(db: Session = Depends(get_db)):
    records = db.query(ImageRecord).all()
    results = []
    
    if os.path.exists("static"):
        files = os.listdir("static")
        image_files = set(f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        
        for record in records:
            if record.filename in image_files:
                results.append({
                    "url": f"http://127.0.0.1:8000/static/{record.filename}",
                    "name": record.filename,
                    "prediction": record.prediction
                })
                
    return results[::-1]

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"static/{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    label, score = predict_image(file_location)
    prediction_text = f"{label} ({score:.1%})"
    
    existing_record = db.query(ImageRecord).filter(ImageRecord.filename == file.filename).first()
    
    if existing_record:
        existing_record.prediction = prediction_text
        existing_record.score = score
    else:
        new_record = ImageRecord(
            filename=file.filename,
            prediction=prediction_text,
            score=score
        )
        db.add(new_record)
    
    db.commit()
    
    return {
        "url": f"http://127.0.0.1:8000/{file_location}", 
        "prediction": prediction_text
    }