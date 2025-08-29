from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from extractor import extract_text  
from classifier import predict       

app = FastAPI(title="Contract Classifier API")

@app.post("/predict")
async def predict_contract(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        label = predict(text)
        return {"filename": file.filename, "predicted_label": label}
    except Exception as e:
        return {"error": str(e)}
