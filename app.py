from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("toxic_model")
model = AutoModelForSequenceClassification.from_pretrained("toxic_model")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    tokens = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=1)
        score = probs[0][1].item()  # assume label 1 = toxic
    return {"toxic": bool(score > 0.5), "score": score}
