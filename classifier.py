# classifier.py
import os
import pickle
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# -------------------
# Config
# -------------------
HF_REPO = "philGyamfi/pgLegalClassification"  
MODEL_FILE = "legalbert.onnx"
TOKENIZER_DIR = "tokenizer"
LABEL_FILE = "le.pkl"

# -------------------
# Ensure model is available
# -------------------
if not os.path.exists(MODEL_FILE):
    print("Downloading ONNX model from Hugging Face...")
    model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
else:
    model_path = MODEL_FILE

# -------------------
# Ensure tokenizer is available
# -------------------
if not os.path.exists(TOKENIZER_DIR):
    print("Downloading tokenizer from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    tokenizer.save_pretrained(TOKENIZER_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# -------------------
# Ensure LabelEncoder is available
# (you can also upload it to HF and download via hf_hub_download if needed)
# -------------------
with open(LABEL_FILE, "rb") as f:
    le = pickle.load(f)

# -------------------
# Load ONNX session
# -------------------
session = ort.InferenceSession(model_path)

# -------------------
# Prediction function
# -------------------
def predict(text: str) -> str:
    """
    Predict the contract type from raw text using the ONNX model.
    """
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="np"
    )

    onnx_input_names = [inp.name for inp in session.get_inputs()]
    onnx_inputs = {k: v for k, v in inputs.items() if k in onnx_input_names}

    # Run inference
    logits = session.run(None, onnx_inputs)[0]
    pred_id = np.argmax(logits, axis=1)[0]

    # Decode
    label = le.inverse_transform([pred_id])[0]
    return label

# -------------------
# Quick test
# -------------------
if __name__ == "__main__":
    sample_text = """
    GENERAL PARTNERSHIP AGREEMENT
    This GENERAL PARTNERSHIP AGREEMENT is made and entered into as of November 5, 2024,
    by and between Eleanor Vance and Julian Santos for operating a specialty coffee roastery and café.
    The partners agree on capital contributions, profit sharing, and management responsibilities.
    """
    print("Predicted label:", predict(sample_text))
