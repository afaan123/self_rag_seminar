import torch
import json
from transformers import DistilBertTokenizerFast
from critique import CriticDistilBERT
from sklearn.model_selection import train_test_split

# -------------------------------
# Load tokenizer and trained model
# -------------------------------
model_path = "model/distilbert"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model = CriticDistilBERT()
model.load_state_dict(torch.load(f"{model_path}/critic_distilbert_best.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------------------
# Reflection Categories
# -------------------------------
CATEGORIES = {
    "Retrieve": ["[Retrieve:yes]", "[Retrieve:no]"],
    "isREL": ["[Relevant]", "[Irrelevant]"],
    "isSUP": ["[Fully supported]", "[Partially supported]", "[No support]"],
    "isUSE": ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
}
def critic_infer_on_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = {
        key: CATEGORIES[key][outputs[key].argmax().item()]
        for key in CATEGORIES
    }
    return preds

def critic_infer_on_passage(instruction, passage, response_segment):
    input_text = f"{instruction}\n{passage}\n{response_segment}"
    preds = critic_infer_on_text(model, tokenizer, input_text)
    return f"{preds['isREL']} {preds['isSUP']}"

def critic_infer_on_final_response(instruction, response):
    input_text = f"{instruction}\n{response}"
    preds = critic_infer_on_text(model, tokenizer, input_text)
    return preds['isUSE']
# -------------------------------
# Load and Split Data
# -------------------------------
with open("data/selfrag_it_critic.json") as f:
    all_data = json.load(f)

_, eval_data = train_test_split(all_data, test_size=0.2, random_state=42)

# -------------------------------
# Evaluation Loop
# -------------------------------
correct = {key: 0 for key in CATEGORIES}
total = len(eval_data)

for i, ex in enumerate(eval_data, 1):
    inputs = tokenizer(ex["input"], return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = {
            "Retrieve": CATEGORIES["Retrieve"][outputs["retrieve"].argmax().item()],
            "isREL": CATEGORIES["isREL"][outputs["isREL"].argmax().item()],
            "isSUP": CATEGORIES["isSUP"][outputs["isSUP"].argmax().item()],
            "isUSE": CATEGORIES["isUSE"][outputs["isUSE"].argmax().item()],
        }

    for key in CATEGORIES:
        if predictions[key] == ex[key]:
            correct[key] += 1

# -------------------------------
# Results
# -------------------------------
print("\n================= Evaluation Results (20%) =================")
for key in CATEGORIES:
    accuracy = correct[key] / total * 100
    print(f"{key} Accuracy: {accuracy:.2f}%")
print("============================================================")

# -------------------------------
# Reflection Token Meanings
# -------------------------------
print("\nReflection Token Meanings:")
print("[Retrieve:yes] - Should retrieve more information")
print("[Retrieve:no] - No need for additional retrieval")
print("[Relevant] - Document is relevant")
print("[Irrelevant] - Document is not relevant")
print("[Fully supported] - Response fully supported by document")
print("[Partially supported] - Response partially supported")
print("[No support] - No support from document")
print("[Utility:1-5] - Response utility (1=low, 5=high)")
