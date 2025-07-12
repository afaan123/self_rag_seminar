import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import re
from tqdm import tqdm

# ------------------ Reflection Tokens ------------------
SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[EOS]", "[Retrieve:yes]", "[Retrieve:no]", "[Retrieve:UNK]",
    "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]", "[No support]",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"
]

# ------------------ Tokenizer ------------------
def build_vocab_from_json(json_data):
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(vocab)
    for sample in json_data:
        for text in [sample['input'], sample['Retrieve'], sample['isREL'], sample['isSUP'], sample['isUSE']]:
            tokens = re.findall(r"\[.*?\]|\w+", text)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
    return vocab

def encode(text, vocab, max_length=64):
    # First, extract all special tokens (anything in brackets)
    special_tokens = re.findall(r"\[.*?\]", text)
    # Then extract regular words, but exclude parts that are already in special tokens
    remaining_text = re.sub(r"\[.*?\]", "", text)
    regular_tokens = re.findall(r"\w+", remaining_text)
    
    # Combine tokens, prioritizing special tokens
    all_tokens = special_tokens + regular_tokens
    return [vocab.get(tok, vocab["[UNK]"]) for tok in all_tokens[:max_length]]

def decode(token_ids, vocab):
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = []
    for tid in token_ids:
        token = id_to_token.get(tid, "[UNK]")
        if token in ["[PAD]", "[EOS]"]:
            continue
        # Keep the token as-is, don't split it
        tokens.append(token)
    return " ".join(tokens)

def decode_tokens(token_ids, vocab):
    """Decode token IDs to a list of tokens (not split by spaces)"""
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = []
    for tid in token_ids:
        token = id_to_token.get(tid, "[UNK]")
        if token in ["[PAD]", "[EOS]"]:
            continue
        tokens.append(token)
    return tokens

# ------------------ Dataset ------------------
class CriticSeqDataset(Dataset):
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

        # Build label using the 4 fields
        label_string = f"{sample['Retrieve']} {sample['isREL']} {sample['isSUP']} {sample['isUSE']} [EOS]"
        label_ids = encode(label_string, self.vocab, self.max_length)

        return {
            'input_ids': input_ids,
            'label_ids': label_ids
        }

def collate_fn(batch):
    max_input_len = max(len(b['input_ids']) for b in batch)
    max_label_len = max(len(b['label_ids']) for b in batch)

    padded_inputs = [b['input_ids'] + [0] * (max_input_len - len(b['input_ids'])) for b in batch]
    padded_labels = [b['label_ids'] + [0] * (max_label_len - len(b['label_ids'])) for b in batch]

    return {
        'input_ids': torch.tensor(padded_inputs, dtype=torch.long),
        'label_ids': torch.tensor(padded_labels, dtype=torch.long)
    }

# ------------------ Seq2Seq Model ------------------
class CriticSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_ids, label_ids):
        src_embeds = self.embed(input_ids)
        enc_out, (h, c) = self.encoder(src_embeds)
        h = torch.cat([h[0], h[1]], dim=-1).unsqueeze(0)
        c = torch.cat([c[0], c[1]], dim=-1).unsqueeze(0)
        tgt_embeds = self.embed(label_ids)
        dec_out, _ = self.decoder(tgt_embeds, (h, c))
        logits = self.fc(dec_out)
        return logits

    def generate(self, input_ids, decoder_input, vocab, max_length=10):
        src_embeds = self.embed(input_ids)
        _, (h, c) = self.encoder(src_embeds)
        h = torch.cat([h[0], h[1]], dim=-1).unsqueeze(0)
        c = torch.cat([c[0], c[1]], dim=-1).unsqueeze(0)

        generated = decoder_input.clone()
        eos_token_id = vocab.get("[EOS]")
        
        # Define the exact tokens we want in order
        expected_tokens = [
            ["[Retrieve:yes]", "[Retrieve:no]"],  # Retrieve category
            ["[Relevant]", "[Irrelevant]"],        # isREL category  
            ["[Fully supported]", "[Partially supported]", "[No support]"],  # isSUP category
            ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]  # isUSE category
        ]
        
        for step in range(4):  # Generate exactly 4 tokens
            tgt_embeds = self.embed(generated)
            dec_out, (h, c) = self.decoder(tgt_embeds, (h, c))
            logits = self.fc(dec_out[:, -1:, :])
            
            # Mask to only allow tokens from the current category
            mask = torch.ones_like(logits) * float('-inf')
            for token in expected_tokens[step]:
                if token in vocab:
                    mask[0, 0, vocab[token]] = 0
            
            logits = logits + mask
            probs = torch.softmax(logits, dim=-1)
            
            # Debug: Print available tokens for this step
            print(f"Step {step + 1} - Available tokens:")
            for token in expected_tokens[step]:
                if token in vocab:
                    prob = probs[0, 0, vocab[token]].item()
                    print(f"  {token}: {prob:.4f}")
            
            next_token = torch.argmax(probs, dim=-1)
            token_text = list(vocab.keys())[list(vocab.values()).index(next_token.item())]
            print(f"Selected: {token_text}")
            
            generated = torch.cat([generated, next_token], dim=1)

        return generated

# ------------------ Training Loop ------------------
def train_critic(model, dataloader, optimizer, loss_fn, epochs=10):
    model.train()
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

# ------------------ Main ------------------
def main():
    json_path = "data/selfrag_it_critic.json"
    with open(json_path) as f:
        json_data = json.load(f)

    vocab = build_vocab_from_json(json_data)
    dataset = CriticSeqDataset(json_path, vocab)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = CriticSeqModel(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    train_critic(model, dataloader, optimizer, loss_fn, epochs=10)

    # Save model and vocab
    save_path = "model/critiqueModel"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "critic_seq_model.pth"))
    with open(os.path.join(save_path, "critic_seq_vocab.json"), "w") as f:
        json.dump(vocab, f)
    print("Model saved.")

if __name__ == "__main__":
    main()
