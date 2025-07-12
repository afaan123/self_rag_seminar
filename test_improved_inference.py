import torch
import json
from critique import CriticSeqModel, encode, decode, decode_tokens, SPECIAL_TOKENS

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

# Generate with improved method
with torch.no_grad():
    decoder_input = torch.tensor([[vocab["[PAD]"]]], dtype=torch.long)
    generated = model.generate(input_tensor, decoder_input, vocab=vocab, max_length=10)

generated_ids = generated[0].tolist()
tokens = decode_tokens(generated_ids, vocab)

print(f"Input: {sample_input}")
print(f"Generated tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Check if we have all categories
categories = {
    "Retrieve": ["[Retrieve:yes]", "[Retrieve:no]"],
    "isREL": ["[Relevant]", "[Irrelevant]"],
    "isSUP": ["[Fully supported]", "[Partially supported]", "[No support]"],
    "isUSE": ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
}

found_categories = set()
for token in tokens:
    for category, valid_tokens in categories.items():
        if token in valid_tokens:
            found_categories.add(category)
            print(f"Found {category}: {token}")

missing = set(categories.keys()) - found_categories
if missing:
    print(f"Missing categories: {missing}")
else:
    print("All categories found!") 