# File: main3.py
# ============================
# MERGED main.py  (Audio + ElevenLabs)
# ============================

import os
import io
import librosa
import numpy as np
import joblib
import tempfile
from dotenv import load_dotenv
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from elevenlabs.client import ElevenLabs
from pydantic import BaseModel
import uvicorn

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# =====================================================================
#                 FASTAPI APP (Shared Between Both Modules)
# =====================================================================
app = FastAPI(
    title="Merged Audio API - Emotion Recognition + Dog SFX",
    description="Includes Emotion Prediction + ElevenLabs Dog Sound Generator",
    version="3.0"
)

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
#                SECTION 1 — AUDIO EMOTION RECOGNITION
# =====================================================================

def extract_features(audio, sample_rate, n_mfcc=40):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


class AudioPredictor:
    def __init__(self, model_path, scaler_path, encoder_path):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoder = joblib.load(encoder_path)
            print("✅ Emotion model loaded successfully.")
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


# Load model once
predictor = AudioPredictor(
    "Model/final_emotion_model.pkl",
    "Model/final_scaler.pkl",
    "Model/final_label_encoder.pkl"
)

@app.post("/predict-audio/")
async def predict_audio(file: UploadFile = File(...)):
    if predictor.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Save uploaded audio
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving uploaded file: {e}")

    # Predict
    prediction = predictor.predict_emotion(tmp_path)
    os.remove(tmp_path)

    if isinstance(prediction, str) and prediction.startswith("Error"):
        raise HTTPException(status_code=400, detail=prediction)

    return {"filename": file.filename, "predicted_emotion": prediction}

# =====================================================================
#                SECTION 2 — ELEVENLABS DOG SOUND GENERATION
# =====================================================================

# Initialize ElevenLabs client
if not ELEVENLABS_API_KEY:
    print("ELEVENLABS_API_KEY not found.")
    client = None
else:
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print("ElevenLabs client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ElevenLabs client: {e}")
        client = None


class BarkRequest(BaseModel):
    prompt: str = Body(..., examples=["a loud bark", "a deep growl"])
    prompt_influence: Optional[float] = Body(
        None,
        examples=[0.4],
        description="Creativity level (0.0 to 1.0)"
    )


@app.post("/generate-bark/")
async def generate_bark(request: BarkRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="ElevenLabs client is not initialized."
        )

    try:
        audio_data = client.text_to_sound_effects.convert(
            text=request.prompt,
            prompt_influence=request.prompt_influence
        )

        if not audio_data:
            raise HTTPException(500, "Audio generation failed.")

        return StreamingResponse(
            audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="bark.mp3"'}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


# =====================================================================
#                 ROOT ENDPOINT
# =====================================================================
@app.get("/")
async def root():
    return {
        "message": "Merged API Running",
        "endpoints": [
            "/predict-audio",
            "/generate-bark"
        ]
    }


# =====================================================================
# RUN SERVER
# =====================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

