# File: main.py
import os
import librosa
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile

# --- Feature Extraction Function ---
def extract_features(audio, sample_rate, n_mfcc=40):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# --- Prediction Pipeline Class ---
class AudioPredictor:
    def __init__(self, model_path, scaler_path, encoder_path):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoder = joblib.load(encoder_path)
            print("✅ Model and objects loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model files: {e}")
            self.model = None

    def predict_emotion(self, file_path: str):
        if self.model is None:
            return "Model not loaded. Cannot make a prediction."

        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        except Exception as e:
            return f"Error loading audio file: {e}"

        features = extract_features(audio, sample_rate)
        if features is None:
            return "Could not extract features from the audio file."

        features_reshaped = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_reshaped)

        prediction_encoded = self.model.predict(features_scaled)
        prediction_label = self.encoder.inverse_transform(prediction_encoded)

        return prediction_label[0]


# --- FastAPI App ---
app = FastAPI(title="Audio Emotion Recognition API")

# Allow CORS if you want frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = "Model/final_emotion_model.pkl"
SCALER_PATH = "Model/final_scaler.pkl"
ENCODER_PATH = "Model/final_label_encoder.pkl"

predictor = AudioPredictor(MODEL_PATH, SCALER_PATH, ENCODER_PATH)


# --- Routes ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Emotion Recognition API 🎶. Visit /docs for Swagger UI."}


@app.get("/favicon.ico")
async def favicon():
    return {}


@app.post("/predict-audio/")
async def predict_audio(file: UploadFile = File(...)):
    if predictor.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving uploaded file: {e}")

    # Predict emotion
    prediction = predictor.predict_emotion(tmp_path)

    # Clean up temp file
    os.remove(tmp_path)

    if isinstance(prediction, str) and prediction.startswith("Error"):
        raise HTTPException(status_code=400, detail=prediction)

    return {"filename": file.filename, "predicted_emotion": prediction}


# Run using: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
