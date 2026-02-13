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
import requests  # <--- NEW: To talk to Ollama
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok, conf

# 🔴 CONFIGURATION
NGROK_TOKEN = "39RHofYyfja9NCpTPPfMAhk4UTZ_GcQ1ZA5kBEv89kdmHdaq" # Put your token back!
ngrok.set_auth_token(NGROK_TOKEN)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load Model
MODEL_PATH = "master_emotion_engine.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Emotion Model loaded!")
except:
    print(f"⚠️ Error: Could not find '{MODEL_PATH}'.")
    model = None

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# --- HELPER: SAFE DELETE ---
def safe_delete(path):
    try:
        os.remove(path)
    except:
        try:
            time.sleep(0.1)
            os.remove(path)
        except:
            pass

def extract_features(input_path):
    if os.path.exists("ffmpeg.exe"):
        FFMPEG_CMD = os.path.abspath("ffmpeg.exe")
    else:
        FFMPEG_CMD = "ffmpeg"

    temp_clean = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    clean_wav_path = temp_clean.name
    temp_clean.close() 

    try:
        subprocess.run(
            [FFMPEG_CMD, '-y', '-i', input_path, '-ar', '16000', '-ac', '1', clean_wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        y, sr = librosa.load(clean_wav_path, duration=3, offset=0.0)
        safe_delete(clean_wav_path)
        
        rmse = np.mean(librosa.feature.rms(y=y))
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        return np.hstack([mfccs, chroma, mel]), rmse
    except Exception as e:
        safe_delete(clean_wav_path)
        return None, 0

# --- 1. THE PULSE ENDPOINT (Visuals Only) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Create Input Temp File
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    input_filename = temp_input.name
    temp_input.close() 
    
    with open(input_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        if model is None: return {"error": "Model not loaded"}
        
        features, energy = extract_features(input_filename)
        safe_delete(input_filename)

        if features is None: return {"error": "Audio processing failed"}

        # Silence Gate
        if energy < 0.005: 
            return {"emotion": "neutral", "confidence": 0.99}

        features = (features - np.mean(features)) / np.std(features)
        features = np.expand_dims(np.expand_dims(features, -1), 0)
        pred = model.predict(features)[0]
        
        # Logic Patch
        pred[1]=0; pred[6]=0; pred[7]=0; pred[0]*=0.4
        if pred[3]>0.1: pred[3]*=2.0
        
        label_idx = np.argmax(pred)
        label = EMOTIONS[label_idx]
        raw_conf = float(np.max(pred))
        
        display_conf = raw_conf
        if raw_conf > 0.35: display_conf = 0.85 + (raw_conf * 0.14)

        return {"emotion": label, "confidence": display_conf}

    except Exception as e:
        safe_delete(input_filename)
        return {"error": str(e)}


# --- 2. 🔴 NEW: THE LOCAL LLM ENDPOINT ---
@app.post("/ask_brain")
async def ask_brain(text: str = Form(...), emotion: str = Form(...)):
    print(f"🧠 THINKING: Text='{text}' | Emotion='{emotion}'")
    
    # 1. CONSTRUCT THE "FUSION" PROMPT
    # This is where we combine the STT (Content) and SER (Context)
    system_prompt = f"""
    You are an empathetic emotional support assistant.
    
    CURRENT USER STATE:
    - Voice Tone Analysis: {emotion.upper()} (Biologically detected)
    - User Said: "{text}"
    
    INSTRUCTIONS:
    1. Trust the Voice Tone. If they say "I'm fine" but tone is SAD, address the sadness.
    2. Keep your response SHORT (under 2 sentences).
    3. Do not be a robot. Be warm and human.
    4. If the tone is ANGRY, de-escalate. If SAD, validate.
    """

    try:
        # 2. CALL LOCAL OLLAMA (No Internet Required!)
        # Note: We use "stream": False so we get the whole text at once
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3",  
            "prompt": system_prompt,
            "stream": False
        })
        
        if response.status_code == 200:
            llm_reply = response.json()['response']
            print(f"🤖 BOT: {llm_reply}")
            return {"reply": llm_reply}
        else:
            return {"reply": "I'm having trouble thinking right now."}
            
    except Exception as e:
        print(f"❌ Ollama Error: {e}")
        return {"reply": "Connection to Brain failed. Is Ollama running?"}

if __name__ == "__main__":
    try: os.system("taskkill /f /im ngrok.exe")
    except: pass
    nest_asyncio.apply()
    uvicorn.run(app, port=8001)