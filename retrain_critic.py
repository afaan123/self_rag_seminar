import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from critique import CriticSeqModel, build_vocab_from_json, encode

class StructuredCriticDataset(Dataset):
    def __init__(self, json_path, vocab, max_length=64):
        with open(json_path) as f:
            self.samples = json.load(f)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = encode(sample['input'], self.vocab, self.max_length)
        
        # Create structured label with all 4 categories
        label_string = f"{sample['Retrieve']} {sample['isREL']} {sample['isSUP']} {sample['isUSE']} [EOS]"
        label_ids = encode(label_string, self.vocab, self.max_length)
        
        return {
            'input_ids': input_ids,
            'label_ids': label_ids
        }

def train_structured_critic():
    json_path = "data/selfrag_it_critic.json"
    with open(json_path) as f:
        json_data = json.load(f)

    vocab = build_vocab_from_json(json_data)
    dataset = StructuredCriticDataset(json_path, vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = CriticSeqModel(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    print("Training structured critic model...")
    model.train()
    for epoch in range(15):  # More epochs for better learning
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids']
            label_ids = batch['label_ids']
            
            logits = model(input_ids, label_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), label_ids.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Save improved model
    save_path = "model/critiqueModel"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "critic_seq_model_improved.pth"))
    with open(os.path.join(save_path, "critic_seq_vocab.json"), "w") as f:
        json.dump(vocab, f)
    print("Improved model saved.")

def collate_fn(batch):
    max_input_len = max(len(b['input_ids']) for b in batch)
    max_label_len = max(len(b['label_ids']) for b in batch)

    padded_inputs = [b['input_ids'] + [0] * (max_input_len - len(b['input_ids'])) for b in batch]
    padded_labels = [b['label_ids'] + [0] * (max_label_len - len(b['label_ids'])) for b in batch]

    return {
        'input_ids': torch.tensor(padded_inputs, dtype=torch.long),
        'label_ids': torch.tensor(padded_labels, dtype=torch.long)
    }

if __name__ == "__main__":
    train_structured_critic() 