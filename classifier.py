import os
import pickle
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# -------------------
# Config
# -------------------
HF_REPO = "philGyamfi/legalbert.onnx"   
MODEL_FILE = "legalbert.onnx"           
TOKENIZER_DIR = "tokenizer"
LABEL_FILE = "../datacleaning/le.pkl"

# -------------------
# Download model + tokenizer
# -------------------
print("⏳ Checking model...")
model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
print("✅ Model ready!")

print("⏳ Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
print("✅ Tokenizer ready!")

print("⏳ Loading label encoder...")
with open(LABEL_FILE, "rb") as f:
    label_encoder = pickle.load(f)
print("✅ Label encoder ready!")

# -------------------
# Inference session
# -------------------
print("⏳ Initializing ONNX session (this may take a minute)...")
session = ort.InferenceSession(model_path)
print("✅ ONNX session ready!")

# -------------------
# Prediction
# -------------------
def predict(text):
    tokens = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Only pass expected inputs
    onnx_input_names = [inp.name for inp in session.get_inputs()]
    onnx_inputs = {k: v for k, v in tokens.items() if k in onnx_input_names}

    outputs = session.run(None, onnx_inputs)
    pred_id = np.argmax(outputs[0], axis=1)[0]
    return label_encoder.inverse_transform([pred_id])[0]


if __name__ == "__main__":
    print("🚀 System ready! You can now run predictions.")
    sample_text = "This AGREEMENT is made and entered into by and between..."
    print("Predicted label:", predict(sample_text))
