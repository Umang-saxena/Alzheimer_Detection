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
ALZHEIMER_CLASSES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented",
]


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


def infer_output_size(output_shape: tuple) -> int | None:
    if not output_shape:
        return None

    last = output_shape[-1]
    if isinstance(last, int):
        return last

    return None


def to_probabilities(output: np.ndarray) -> np.ndarray:
    # Convert logits to probabilities when needed.
    if np.any(output < 0) or not np.isclose(float(np.sum(output)), 1.0, atol=1e-2):
        shifted = output - np.max(output)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    return output


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
        category = "Demented" if score >= 0.5 else "Non_Demented"
        return {"category": category, "confidence": score}

    if output.size != len(ALZHEIMER_CLASSES):
        raise ValueError(
            "Incompatible model output. Expected 4 Alzheimer classes, "
            f"but got {output.size}."
        )

    probabilities = to_probabilities(output.astype(np.float64))
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    category = ALZHEIMER_CLASSES[index]

    return {
        "category": category,
        "class_index": index,
        "confidence": confidence,
        "class_probabilities": {
            ALZHEIMER_CLASSES[i]: float(probabilities[i])
            for i in range(len(ALZHEIMER_CLASSES))
        },
    }


@app.on_event("startup")
def startup_event() -> None:
    get_model()


@app.get("/")
def root() -> dict:
    return {"message": "Alzheimer detection API is running"}


@app.get("/health")
def health() -> dict:
    model = get_model()
    output_size = infer_output_size(model.output_shape)
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "model_output_size": output_size,
        "alzheimers_compatible": output_size in (1, len(ALZHEIMER_CLASSES)),
    }


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

    try:
        formatted = format_prediction(prediction)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "hint": "Load your Alzheimer-trained model with 4 output classes (or binary output).",
            },
        ) from exc

    return {
        "filename": file.filename,
        "prediction": formatted,
    }