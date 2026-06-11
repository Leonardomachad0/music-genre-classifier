from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db, Prediction
from app.schemas import PredictionResponse, HealthResponse
from src.features import extract_features
import joblib
import numpy as np
import tempfile
import os

router = APIRouter()

# carrega o modelo, scaler e label encoder uma vez só quando a API sobe
MODEL_PATH = "model/xgb_music_classifier.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    model_loaded = True
except Exception as e:
    print(f"erro ao carregar modelo: {e}")
    model_loaded = False


@router.get("/health", response_model=HealthResponse)
def health_check():
    # verifica se a API está no ar e se o modelo foi carregado
    return {"status": "ok", "model_loaded": model_loaded}


@router.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="modelo não disponível")

    # salva o arquivo temporariamente para o librosa conseguir ler
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        # extrai features, normaliza e prediz
        features = extract_features(tmp_path)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(probabilities.max())
        genre = label_encoder.inverse_transform([prediction])[0]

        # loga no banco
        record = Prediction(
            filename=file.filename,
            genre_predicted=genre,
            confidence=confidence
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return PredictionResponse(
            genre=genre,
            confidence=confidence,
            filename=file.filename,
            created_at=record.created_at
        )

    finally:
        # apaga o arquivo temporário
        os.unlink(tmp_path)


@router.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    # retorna o histórico de predições
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    return predictions