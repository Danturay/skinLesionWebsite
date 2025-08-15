from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array 
import numpy as np 
from PIL import Image 
import io
import os 
import cv2


app = FastAPI(title="Skin Lesion Classification API")

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",  # sometimes needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 124
CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]  # your labels


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where mainAPI.py is
MODEL_PATH = os.path.join(BASE_DIR, "baseSmoteFocal.h5")  # full path to model

model = load_model(MODEL_PATH, compile=False)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()

        # Open, convert, and resize
        image = Image.open(io.BytesIO(contents)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        image = np.asarray(image).astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)  # Shape: (1, IMG_SIZE, IMG_SIZE, 3)

        # Predict
        preds = model.predict(image)[0]  # (7,) for 7 classes

        # Get top 3 predictions
        top3_idx = preds.argsort()[-3:][::-1]  # indices of top 3
        top3 = [{"class": CLASS_NAMES[i], "confidence": float(preds[i])} for i in top3_idx]

        return JSONResponse({"top3_predictions": top3})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
async def root():
    return {"message": "Skin Lesion Classification API. Use POST /predict with an image file."}