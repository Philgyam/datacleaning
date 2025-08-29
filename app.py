from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from extractor import extract_text
from classifier import predict
from contextlib import asynccontextmanager
import threading
import webbrowser

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    threading.Timer(1, lambda: webbrowser.open("http://127.0.0.1:8000/docs")).start()
    yield
  

app = FastAPI(title="Contract Classifier API", lifespan=lifespan)

@app.post("/predict")
async def predict_contract(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        label = predict(text)
        return {"filename": file.filename, "predicted_label": label}
    except Exception as e:
        return {"error": str(e)}
