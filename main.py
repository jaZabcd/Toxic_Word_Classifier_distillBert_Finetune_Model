from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Reload from folder
tokenizer = AutoTokenizer.from_pretrained("toxic_model")
model = AutoModelForSequenceClassification.from_pretrained("toxic_model")

# Create pipeline
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)  # device=0 if GPU, -1 for CPU

# Label mapping
id2label = {0: "non-toxic", 1: "toxic"}  # <-- adjust to your dataset

# Run prediction
pred = pipe("fuck off bitch")[0]

# Map label
pred["label"] = id2label[int(pred["label"].split("_")[-1])]
print(pred)
