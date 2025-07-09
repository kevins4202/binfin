import os
import uuid
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
import json
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import base64
from io import BytesIO
import shutil
from load_classifier import load_classifier
import torchaudio
import tempfile

load_dotenv()

app = Flask(__name__, static_folder="static/dist", static_url_path="")
CORS(app)

# Load model with proper error handling
try:
    print("Loading model...")
    model = load_classifier(freeze_beats=True, load_base=True, load_head=True)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/predict", methods=["POST"])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500
    
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if file is a .wav file
        if not file.filename.lower().endswith('.wav'):
            return jsonify({"error": "Only .wav files are supported"}), 400
        
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(temp_file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if necessary (BEATs expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Normalize audio
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Add batch dimension if not present
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Move to device
            waveform = waveform.to(device)
            
            # Create padding mask (all False since we're not padding)
            padding_mask = torch.zeros(waveform.shape[0], waveform.shape[1], dtype=torch.bool, device=device)
            
            # Run inference
            with torch.no_grad():
                logits = model(waveform, padding_mask)
                probabilities = torch.softmax(logits, dim=1)
                
                # Get prediction
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
                
                # Map prediction to label (0: real, 1: fake)
                label = "fake" if prediction == 1 else "real"
                
                return jsonify({
                    "prediction": label,
                    "confidence": confidence,
                    "probabilities": {
                        "real": probabilities[0][0].item(),
                        "fake": probabilities[0][1].item()
                    }
                })
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device if 'device' in locals() else "unknown"
    })

if __name__ == "__main__":
    print("Starting Flask server...")
    print(f"Server will be available at: http://localhost:5015")
    print(f"Health check: http://localhost:5015/api/health")
    print(f"Prediction endpoint: http://localhost:5015/api/predict")
    app.run(host="0.0.0.0", port=5015, debug=True)
