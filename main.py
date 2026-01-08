import os
import shutil
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

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

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- App Setup ---
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
        print("✅ Modell erfolgreich geladen.")
        return model
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        return None

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# --- Endpoints ---

@app.get("/images")
def get_images(db: Session = Depends(get_db)):
    # Lade alle Einträge aus der Datenbank
    records = db.query(ImageRecord).all()
    results = []
    
    # Prüfe physisch vorhandene Dateien und matche sie mit DB-Einträgen
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
                
    return results[::-1] # Neueste zuerst

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"static/{file.filename}"
    
    # 1. Datei speichern
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. KI Vorhersage
    label, score = predict_image(file_location)
    prediction_text = f"{label} ({score:.1%})"
    
    # 3. Datenbank Update oder Insert
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