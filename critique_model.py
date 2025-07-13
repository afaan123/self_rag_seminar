import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
# ---------------------
# Hyperparameters
# ---------------------
BATCH_SIZE = 8
EPOCHS = 20
MAX_LEN = 256
DEVICE = torch.device("cpu")  # Use CPU

# ---------------------
# Label spaces
# ---------------------
CATEGORIES = {
    "Retrieve": ["[Retrieve:yes]", "[Retrieve:no]"],
    "isREL": ["[Relevant]", "[Irrelevant]"],
    "isSUP": ["[Fully supported]", "[Partially supported]", "[No support]"],
    "isUSE": ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
}

label_encoders = {k: LabelEncoder().fit(v) for k, v in CATEGORIES.items()}

# ---------------------
# Dataset
# ---------------------
class CriticDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path) as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["input"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        def label_idx(category, value):
            return CATEGORIES[category].index(value)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "retrieve": label_idx("Retrieve", sample["Retrieve"]),
            "isREL": label_idx("isREL", sample["isREL"]),
            "isSUP": label_idx("isSUP", sample["isSUP"]),
            "isUSE": label_idx("isUSE", sample["isUSE"]),
        }

    @classmethod
    def from_list(cls, samples, tokenizer):
        dataset = cls.__new__(cls)
        dataset.samples = samples
        dataset.tokenizer = tokenizer
        return dataset
# ---------------------
# Model
# ---------------------
class CriticDistilBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.encoder.config.hidden_size

        self.retrieve_head = nn.Linear(hidden_size, 2)
        self.isREL_head = nn.Linear(hidden_size, 2)
        self.isSUP_head = nn.Linear(hidden_size, 3)
        self.isUSE_head = nn.Linear(hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token

        return {
            "retrieve": self.retrieve_head(cls_output),
            "isREL": self.isREL_head(cls_output),
            "isSUP": self.isSUP_head(cls_output),
            "isUSE": self.isUSE_head(cls_output)
        }
# ---------------------
# Training Loop
# ---------------------

def train():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = CriticDataset("data/selfrag_it_critic.json", tokenizer)

    # 80-20 split
    split_idx = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(split_idx))
    val_set = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = CriticDistilBERT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Optional: scheduler
    # total_steps = len(train_loader) * EPOCHS
    # scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float("inf")
    patience = 3
    stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids, attention_mask)

            loss = (
                loss_fn(outputs["retrieve"], batch["retrieve"].to(DEVICE)) +
                loss_fn(outputs["isREL"], batch["isREL"].to(DEVICE)) +
                loss_fn(outputs["isSUP"], batch["isSUP"].to(DEVICE)) +
                loss_fn(outputs["isUSE"], batch["isUSE"].to(DEVICE))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()  # optional
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = {"retrieve": 0, "isREL": 0, "isSUP": 0, "isUSE": 0}
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                targets = {k: batch[k].to(DEVICE) for k in correct.keys()}

                outputs = model(input_ids, attention_mask)

                val_loss += (
                    loss_fn(outputs["retrieve"], targets["retrieve"]) +
                    loss_fn(outputs["isREL"], targets["isREL"]) +
                    loss_fn(outputs["isSUP"], targets["isSUP"]) +
                    loss_fn(outputs["isUSE"], targets["isUSE"])
                ).item()

                # Accuracy
                for key in correct.keys():
                    pred = outputs[key].argmax(dim=1)
                    correct[key] += (pred == targets[key]).sum().item()
                total += input_ids.size(0)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        print("Validation Accuracies:")
        for k in correct:
            print(f"  {k}: {correct[k]/total:.2%}")

        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stop_counter = 0
            torch.save(model.state_dict(), "model/distilbert/critic_distilbert_best.pth")
            print("Best model saved.")
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break


if __name__ == "__main__":
    train()
