
import nest_asyncio, uvicorn, shutil, os, librosa, numpy as np, tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = tf.keras.models.load_model('master_emotion_engine.h5')

# PASTE TOKEN HERE WHEN RUNNING LOCALLY
ngrok.set_auth_token("YOUR_TOKEN_HERE") 

def extract_features(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    
    # NEW: Standardize the audio volume before extracting
    y = librosa.util.normalize(y) 
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    
    # NEW: Z-score normalization (Centers the data)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    
    return mfcc

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as f: shutil.copyfileobj(file.file, f)
    feat = extract_features("temp.wav")
    feat = (feat - np.mean(feat)) / np.std(feat)
    pred = model.predict(np.expand_dims(np.expand_dims(feat, -1), 0))[0]
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    return {"emotion": emotions[np.argmax(pred)], "confidence": float(np.max(pred))}

if __name__ == "__main__":
    ngrok.kill()
    print(f"URL: {ngrok.connect(8000).public_url}/predict")
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
