import json
from typing import List
import torch
from transformers import DistilBertTokenizerFast
from critique import CriticDistilBERT

# ----------- Load your Critic model and tokenizer -----------
model_path = "model/distilbert"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model = CriticDistilBERT()
model.load_state_dict(torch.load(f"{model_path}/critic_distilbert_best.pth", map_location=torch.device("cpu")))
model.eval()

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

    preds = {}
    # Map lowercase output keys to uppercase CATEGORIES keys
    key_map = {
        'retrieve': 'Retrieve',
        'isREL': 'isREL',
        'isSUP': 'isSUP',
        'isUSE': 'isUSE'
    }

    for out_key, cat_key in key_map.items():
        logits = outputs[out_key]  # tensor
        pred_idx = logits.argmax(dim=1).item()  # get predicted class index
        preds[cat_key] = CATEGORIES[cat_key][pred_idx]

    return preds



def critic_infer_on_passage(instruction, passage, response_segment):
    input_text = f"{instruction}\n{passage}\n{response_segment}"
    preds = critic_infer_on_text(model, tokenizer, input_text)
    return f"{preds['isREL']} {preds['isSUP']}"

def critic_infer_on_final_response(instruction, response):
    input_text = f"{instruction}\n{response}"
    preds = critic_infer_on_text(model, tokenizer, input_text)
    return preds['isUSE']

# ----------- Load corpus -----------
def load_corpus(path="data/corpus.txt") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

corpus = load_corpus()

# ----------- Simple Retriever over corpus -----------
def retrieve_top_k_passages(query: str, top_k: int = 3) -> List[str]:
    query_words = set(query.lower().split())
    scored = []
    for doc in corpus:
        doc_words = set(doc.lower().split())
        score = len(query_words.intersection(doc_words))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored if score > 0][:top_k]
    if len(top_docs) < top_k:
        for doc in corpus:
            if doc not in top_docs:
                top_docs.append(doc)
            if len(top_docs) == top_k:
                break
    return top_docs

# ----------- Load your critique dataset -----------
with open("data/selfrag_it_critic.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

output_path = "data/selfrag_generator_train_augmented.jsonl"
with open(output_path, "w", encoding="utf-8") as out_f:
    for example in dataset:
        input_text = example.get("input", "")
        parts = input_text.split("### Response:")
        if len(parts) != 2:
            print("Skipping malformed example:", input_text[:50])
            continue
        instruction_doc = parts[0].strip()
        response_text = parts[1].strip()

        # Retrieve passages
        retrieved_passages = retrieve_top_k_passages(instruction_doc, top_k=3)

        # Critic inference on retrieved passages
        passages_with_reflections = []
        for passage in retrieved_passages:
            reflection_tokens = critic_infer_on_passage(instruction_doc, passage, response_text)
            wrapped = f"<p> {passage} </p> {reflection_tokens}"
            passages_with_reflections.append(wrapped)

        # Critic inference on final response
        utility_token = critic_infer_on_final_response(instruction_doc, response_text)

        # Compose final augmented example
        generator_input = (
            f"{instruction_doc}\n"
            + "\n".join(passages_with_reflections) + "\n"
            + f"[Retrieve:yes] {response_text} {utility_token}"
        )

        # Save as JSONL line
        json_line = json.dumps({"text": generator_input}, ensure_ascii=False)
        out_f.write(json_line + "\n")

print(f"Generator training data saved to {output_path}")
