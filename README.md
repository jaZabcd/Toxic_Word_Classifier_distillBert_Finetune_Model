# Toxic Word Classification 🧹

This project fine-tunes a Hugging Face Transformer model to classify text into **toxic** and **non-toxic** categories.  
It is built with **PyTorch** and **Transformers**.

---

## 🔧 Overview

The workflow involves:

1. **Dataset Preparation**
   - Collect a labeled dataset of text with toxic / non-toxic labels (e.g., [Jigsaw Toxic Comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)).
   - Preprocess text (clean, lowercase, remove unnecessary symbols if required).

2. **Model Selection**
   - Use a pre-trained Transformer model (e.g., `distilbert-base-uncased`).
   - Fine-tune it for binary classification.

3. **Training**
   - Tokenize input text using Hugging Face `AutoTokenizer`.
   - Train the model (`AutoModelForSequenceClassification`) with a binary cross-entropy / softmax loss.
   - Use AdamW optimizer and learning rate scheduler.

4. **Evaluation**
   - Evaluate on a validation set.
   - Track metrics like accuracy, precision, recall, and F1 score.

5. **Saving the Model**
   - Save the trained model + tokenizer locally using `model.save_pretrained("toxic_model")` and `tokenizer.save_pretrained("toxic_model")`.

6. **Prediction / Inference**
   - Reload model and tokenizer.
   - Run inference using Hugging Face `TextClassificationPipeline`.
   - Map predictions (`LABEL_0`, `LABEL_1`) to human-readable classes:  
     - `LABEL_0 → non-toxic`  
     - `LABEL_1 → toxic`

---


---

## 🛠 Tech Stack
- [Python 3.10+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Torchvision](https://pytorch.org/vision/stable/) (optional, for extra datasets/utils)

---

## 📤 Example Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("toxic_model")
model = AutoModelForSequenceClassification.from_pretrained("toxic_model")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)

id2label = {0: "non-toxic", 1: "toxic"}

pred = pipe("You are stupid!")[0]
pred["label"] = id2label[int(pred["label"].split("_")[-1])]
print(pred)


## 📂 Workflow

