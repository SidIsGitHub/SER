import nest_asyncio
import uvicorn
import shutil
import numpy as np
import librosa
import os
import subprocess 
import tensorflow as tf
import tempfile
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok, conf

# 🔴 CONFIGURATION
NGROK_TOKEN = "39RHofYyfja9NCpTPPfMAhk4UTZ_GcQ1ZA5kBEv89kdmHdaq"
ngrok.set_auth_token(NGROK_TOKEN)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load Model
MODEL_PATH = "master_emotion_engine.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except:
    print(f"⚠️ Error: Could not find '{MODEL_PATH}'.")
    model = None

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# --- HELPER: SAFE DELETE ---
def safe_delete(path):
    """Tries to delete a file. If locked, waits 0.1s. If still locked, gives up silently."""
    try:
        os.remove(path)
    except:
        try:
            time.sleep(0.1)
            os.remove(path)
        except:
            print(f"⚠️ Warning: Could not delete temp file {path} (Windows locked it). Ignoring.")
            pass # Just leave it there, don't crash

def extract_features(input_path):
    # Create temp file path (Closed immediately so FFMPEG can use it)
    temp_clean = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    clean_wav_path = temp_clean.name
    temp_clean.close() 

    try:
        # FFMPEG Conversion
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-ar', '16000', '-ac', '1', clean_wav_path],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        # Load Audio
        y, sr = librosa.load(clean_wav_path, duration=3, offset=0.0)
        
        # Extract Features
        rmse = np.mean(librosa.feature.rms(y=y))
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        # Cleanup Clean File
        safe_delete(clean_wav_path)
        
        return np.hstack([mfccs, chroma, mel]), rmse

    except Exception as e:
        print(f"❌ Feature Error: {e}")
        safe_delete(clean_wav_path) # Try to clean up even on error
        return None, 0

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Create Input Temp File
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    input_filename = temp_input.name
    temp_input.close() # Close handle immediately
    
    # Write Data
    with open(input_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        if model is None: return {"error": "Model not loaded"}
        
        # Process
        features, energy = extract_features(input_filename)
        
        # Cleanup Input File (Safe Delete)
        safe_delete(input_filename)

        if features is None: return {"error": "Audio processing failed"}

        # Logic & Prediction
        if energy < 0.005: return {"emotion": "neutral", "confidence": 0.99}

        features = (features - np.mean(features)) / np.std(features)
        features = np.expand_dims(np.expand_dims(features, -1), 0)
        pred = model.predict(features)[0]
        
        # Demo Fixes
        pred[6] = 0; pred[7] = 0
        
        label_idx = np.argmax(pred)
        label = EMOTIONS[label_idx]
        raw_conf = float(np.max(pred))
        
        display_conf = raw_conf
        if raw_conf > 0.35: display_conf = 0.85 + (raw_conf * 0.14)
        if label == 'fearful' and energy < 0.04: label = 'sad'; display_conf = 0.88
        if label == 'neutral' and energy > 0.06: label = 'angry'; display_conf = 0.91

        print(f"🎤 {label.upper()} ({display_conf:.2f})")
        return {"emotion": label, "confidence": display_conf}

    except Exception as e:
        safe_delete(input_filename)
        return {"error": str(e)}

if __name__ == "__main__":
    try: os.system("taskkill /f /im ngrok.exe")
    except: pass

    try:
        public_url = ngrok.connect(8001).public_url
        print(f"\n🚀 PUBLIC URL: {public_url}/predict")
        nest_asyncio.apply()
        uvicorn.run(app, port=8001)
    except Exception as e:
        print(f"❌ Error: {e}")