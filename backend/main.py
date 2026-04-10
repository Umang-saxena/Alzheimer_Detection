from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input


MODEL_PATH = Path(__file__).with_name("resnet50.keras")
IMAGE_SIZE = (224, 224)


app = FastAPI(title="Alzheimer Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model() -> keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    return keras.models.load_model(MODEL_PATH, compile=False)


def prepare_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)


def format_prediction(prediction: np.ndarray) -> dict:
    output = np.asarray(prediction)

    if output.ndim == 0:
        score = float(output)
        label = "demented" if score >= 0.5 else "not_demented"
        return {"label": label, "score": score, "raw": score}

    if output.ndim > 1:
        output = output[0]

    if output.size == 1:
        score = float(output[0])
        label = "demented" if score >= 0.5 else "not_demented"
        return {"label": label, "score": score, "raw": [score]}

    class_names = [
        "Non_Demented",
        "Very_Mild_Demented",
        "Mild_Demented",
        "Moderate_Demented",
    ]
    index = int(np.argmax(output))
    confidence = float(output[index])
    label = class_names[index] if index < len(class_names) else f"class_{index}"

    return {
        "label": label,
        "class_index": index,
        "confidence": confidence,
        "raw": [float(value) for value in output.tolist()],
    }


@app.on_event("startup")
def startup_event() -> None:
    get_model()


@app.get("/")
def root() -> dict:
    return {"message": "Alzheimer detection API is running"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload")

    model = get_model()
    prepared = prepare_image(image_bytes)
    prediction = model.predict(prepared, verbose=0)

    return {
        "filename": file.filename,
        "prediction": format_prediction(prediction),
    }