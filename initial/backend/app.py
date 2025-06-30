# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import numpy as np
from transformers import AutoModel
import xgboost as xgb
import torch.nn as nn
from pathlib import Path
import tempfile
import os
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WavLM Deepfake Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    real_probability: float
    fake_probability: float

# WavLM Feature Extractor (same as your training code)
class WavLMFeatureExtractor(nn.Module):
    def __init__(self, model, apply_pooling=True):
        super().__init__()
        self.wavlm = model
        self.apply_pooling = apply_pooling
        if apply_pooling:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Freeze the parameters of the WavLM model
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, input_values):
        outputs = self.wavlm(input_values.squeeze(1), output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state

        if self.apply_pooling:
            pooled_features = self.avg_pool(last_hidden_state.permute(0, 2, 1))
            return pooled_features.squeeze(-1)
        else:
            return last_hidden_state

# Global variables for models
feature_extractor = None
xgb_model = None
device = None
TARGET_SAMPLE_RATE = 16000

def load_models():
    """Load the WavLM and XGBoost models"""
    global feature_extractor, xgb_model, device
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load WavLM model
        logger.info("Loading WavLM model...")
        wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base")
        feature_extractor = WavLMFeatureExtractor(wavlm_model, apply_pooling=True)
        feature_extractor.to(device)
        feature_extractor.eval()
        
        # Load XGBoost model
        logger.info("Loading XGBoost model...")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("xgb_model.json")  # Make sure this file exists
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def preprocess_audio(file_path: str):
    """Preprocess audio file for inference"""
    try:
        # Load audio file
        waveform, original_sample_rate = torchaudio.load(file_path)
        
        # Resample if necessary
        if original_sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")

def extract_features(waveform):
    """Extract features using WavLM"""
    try:
        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = feature_extractor(waveform)
        
        return features.cpu().numpy()
    
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    load_models()

@app.get("/")
async def root():
    return {"message": "WavLM Deepfake Detection API", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict if an uploaded audio file is real or fake
    """
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    if feature_extractor is None or xgb_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Preprocess audio
            waveform = preprocess_audio(temp_file.name)
            
            # Extract features
            features = extract_features(waveform)
            
            # Make prediction
            prediction_proba = xgb_model.predict_proba(features)[0]
            prediction_class = xgb_model.predict(features)[0]
            
            # Calculate probabilities
            fake_prob = float(prediction_proba[0])  # Class 0 = fake
            real_prob = float(prediction_proba[1])  # Class 1 = real
            
            # Determine prediction and confidence
            if prediction_class == 1:
                prediction = "REAL"
                confidence = real_prob
            else:
                prediction = "FAKE"
                confidence = fake_prob
            
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                real_probability=real_prob,
                fake_probability=fake_prob
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)