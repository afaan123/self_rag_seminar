import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# === Dataset with masking for retrieved passages ===
class SelfRAGGeneratorDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # Tokenize with truncation & padding
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # pad here to max_length
            return_tensors=None,   # returns dict of lists, not tensors
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        labels = input_ids.copy()

        p_start = self.tokenizer.convert_tokens_to_ids("<p>")
        p_end = self.tokenizer.convert_tokens_to_ids("</p>")

        inside_p = False
        for i, token_id in enumerate(input_ids):
            if token_id == p_start:
                inside_p = True
                labels[i] = -100
            elif token_id == p_end:
                inside_p = False
                labels[i] = -100
            elif inside_p:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# === Load corpus ===
def load_corpus(path="data/corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

corpus = load_corpus()

# === Retriever function ===
def retrieve_passages(query, top_k=3):
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

# === Dummy Critic scoring (replace with your real critic) ===
def critic_score(generated_text, candidate_text, passage):
    return len(candidate_text)

# === Adaptive retrieval + tree decoding generation ===
def generate_with_adaptive_retrieval(
    model,
    tokenizer,
    prompt,
    retriever,
    critic,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_tokens=256,
    retrieve_threshold=0.5,
    top_k=3,
    beam_size=2,
):
    model.eval()
    model.to(device)

    generated_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_text = prompt

    cur_len = generated_tokens.size(1)

    while cur_len < max_tokens:
        outputs = model(generated_tokens)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)

        retrieve_yes_token_id = tokenizer.convert_tokens_to_ids("[Retrieve:yes]")
        retrieve_yes_prob = probs[0, retrieve_yes_token_id].item() if retrieve_yes_token_id != tokenizer.unk_token_id else 0.0

        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
        next_token_str = tokenizer.decode(next_token_id[0].tolist())
        generated_text += next_token_str
        cur_len += 1

        if retrieve_yes_prob > retrieve_threshold:
            retrieved_passages = retriever(generated_text, top_k=top_k)

            candidates = []
            for passage in retrieved_passages:
                passage_prompt = generated_text + f"\n<p> {passage} </p>"
                input_ids = tokenizer(passage_prompt, return_tensors="pt").input_ids.to(device)

                candidate_ids = model.generate(
                    input_ids, max_new_tokens=32, num_beams=beam_size
                )
                candidate_text = tokenizer.decode(
                    candidate_ids[0][input_ids.size(1) :], skip_special_tokens=True
                )
                score = critic(generated_text, candidate_text, passage)
                candidates.append((candidate_text, score))

            best_candidate = max(candidates, key=lambda x: x[1])[0]

            generated_text += best_candidate
            new_tokens = tokenizer(best_candidate, return_tensors="pt").input_ids.to(device)
            generated_tokens = torch.cat([generated_tokens, new_tokens], dim=1)
            cur_len += new_tokens.size(1)

        if next_token_str.strip() == tokenizer.eos_token:
            break

    return generated_text


def main():
    model_name_or_path = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    special_tokens = {
        "additional_special_tokens": [
            "[Retrieve:yes]",
            "[Retrieve:no]",
            "[Relevant]",
            "[Irrelevant]",
            "[Fully supported]",
            "[Partially supported]",
            "[No support]",
            "[Utility:1]",
            "[Utility:2]",
            "[Utility:3]",
            "[Utility:4]",
            "[Utility:5]",
            "<p>",
            "</p>",
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    dataset = SelfRAGGeneratorDataset("data/selfrag_generator_train_augmented.jsonl", tokenizer)

    training_args = TrainingArguments(
        output_dir="./generator_finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model("./generator_finetuned")
    tokenizer.save_pretrained("./generator_finetuned")
    print("Training complete and model saved.")

    # Example inference prompt
    prompt = "### Instruction: What is cloud computing?"
    generated = generate_with_adaptive_retrieval(
        model, tokenizer, prompt, retrieve_passages, critic_score
    )
    print("Generated output:\n", generated)


if __name__ == "__main__":
    main()
