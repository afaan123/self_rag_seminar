import torch
import json
from critique_model import CriticSeqModel, encode, decode, SPECIAL_TOKENS

# Load model and vocab
model_path = "model/critiqueModel/critic_seq_model.pth"
vocab_path = "model/critiqueModel/critic_seq_vocab.json"

with open(vocab_path) as f:
    vocab = json.load(f)

model = CriticSeqModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=256)
model.load_state_dict(torch.load(model_path))
model.eval()

# Test input
sample_input = """### Instruction: What is an API?
### Document: APIs allow different systems to communicate.
### Response: An API lets software talk to each other."""

input_ids = encode(sample_input, vocab)
input_tensor = torch.tensor([input_ids], dtype=torch.long)

print("Expected tokens in vocabulary:")
expected_tokens = [
    ["[Retrieve:yes]", "[Retrieve:no]"],
    ["[Relevant]", "[Irrelevant]"],
    ["[Fully supported]", "[Partially supported]", "[No support]"],
    ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
]

for i, category in enumerate(["Retrieve", "isREL", "isSUP", "isUSE"]):
    print(f"{category}:")
    for token in expected_tokens[i]:
        if token in vocab:
            print(f"  {token}: {vocab[token]}")
        else:
            print(f"  {token}: NOT IN VOCAB")

print("\n" + "="*50)

# Test generation step by step
with torch.no_grad():
    decoder_input = torch.tensor([[vocab["[PAD]"]]], dtype=torch.long)
    generated = decoder_input.clone()
    
    src_embeds = model.embed(input_tensor)
    _, (h, c) = model.encoder(src_embeds)
    h = torch.cat([h[0], h[1]], dim=-1).unsqueeze(0)
    c = torch.cat([c[0], c[1]], dim=-1).unsqueeze(0)
    
    for step in range(4):
        tgt_embeds = model.embed(generated)
        dec_out, (h, c) = model.decoder(tgt_embeds, (h, c))
        logits = model.fc(dec_out[:, -1:, :])
        
        # Show top 5 tokens for this step
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs.squeeze(), 5)
        
        print(f"\nStep {step + 1} - Top 5 tokens:")
        for prob, idx in zip(top_probs, top_indices):
            token = list(vocab.keys())[list(vocab.values()).index(idx.item())]
            print(f"  {token}: {prob:.4f}")
        
        # Apply masking for expected category
        mask = torch.ones_like(logits) * float('-inf')
        for token in expected_tokens[step]:
            if token in vocab:
                mask[0, 0, vocab[token]] = 0
        
        logits = logits + mask
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        
        token_text = list(vocab.keys())[list(vocab.values()).index(next_token.item())]
        print(f"Selected token: {token_text}")
        
        generated = torch.cat([generated, next_token], dim=1)

print(f"\nFinal generated sequence:")
generated_ids = generated[0].tolist()
decoded = decode(generated_ids, vocab)
tokens = decoded.split()
print(f"Tokens: {tokens}") 