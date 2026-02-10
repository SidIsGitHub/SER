import nest_asyncio
import uvicorn
import shutil
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import tensorflow as tf


NGROK_TOKEN = "39RHofYyfja9NCpTPPfMAhk4UTZ_GcQ1ZA5kBEv89kdmHdaq"
ngrok.set_auth_token(NGROK_TOKEN)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load Model
model = tf.keras.models.load_model('master_emotion_engine.h5')

# EMOTION LIST (Note: Index 6 is Disgust)
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    
    # ENERGY CHECK (Volume)
    rmse = np.mean(librosa.feature.rms(y=y))
    
    # Feature Extraction
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    return np.hstack([mfccs, chroma, mel]), rmse

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = "temp_server.wav"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        features, energy = extract_features(filename)
        
        # 1. SILENCE CHECK: If audio is near silent, force Neutral
        if energy < 0.01:
            return {"emotion": "neutral", "confidence": 1.0, "note": "Audio too quiet"}

        # Normalize
        features = (features - np.mean(features)) / np.std(features)
        features = np.expand_dims(np.expand_dims(features, -1), 0)
        
        # Predict
        pred = model.predict(features)[0]
        
        # 🔴 THE FIX: KILL DISGUST
        # Set Disgust (Index 6) to 0 and re-normalize
        pred[6] = 0 
        pred = pred / np.sum(pred) # Make percentages add up to 100% again

        # Get final result
        label_idx = np.argmax(pred)
        label = EMOTIONS[label_idx]
        confidence = float(np.max(pred))
        
        # 🟡 SECONDARY FIX: If it's "Fearful" but quiet, it's probably Sad
        if label == 'fearful' and energy < 0.04:
            label = 'sad'

        print(f"🎤 Heard: {label.upper()} ({confidence:.2f}) | Energy: {energy:.4f}")
        return {"emotion": label, "confidence": confidence}
        
    except Exception as e:
        return {"error": str(e)}

# Start Server
ngrok.kill()
public_url = ngrok.connect(8000).public_url
print(
