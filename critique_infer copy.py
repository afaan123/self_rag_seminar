import torch
import json
from critique import CriticSeqModel, encode, decode, decode_tokens, SPECIAL_TOKENS

# -------------------------------
# Define reflection token categories
# -------------------------------
REFLECTION_CATEGORIES = {
    "Retrieve": ["[Retrieve:yes]", "[Retrieve:no]"],
    "isREL": ["[Relevant]", "[Irrelevant]"],
    "isSUP": ["[Fully supported]", "[Partially supported]", "[No support]"],
    "isUSE": ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
}


def extract_reflection_fields(tokens):
    """Extract first token per category from generated output"""
    results = {k: None for k in REFLECTION_CATEGORIES.keys()}

    for tok in tokens:
        for key, valid_tokens in REFLECTION_CATEGORIES.items():
            if results[key] is None and tok in valid_tokens:
                results[key] = tok

    # Set defaults for any missing categories
    for key in results:
        if results[key] is None:
            results[key] = "[UNK]"

    return results


# -------------------------------
# Load model and vocab
# -------------------------------
model_path = "model/critiqueModel/critic_seq_model.pth"
vocab_path = "model/critiqueModel/critic_seq_vocab.json"

with open(vocab_path) as f:
    vocab = json.load(f)

model = CriticSeqModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=256)
model.load_state_dict(torch.load(model_path))
model.eval()

# -------------------------------
# Test Inputs (can load from dataset)
# -------------------------------
test_inputs = [
    {
        "input": """### Instruction: What is an API?
### Document: APIs allow different systems to communicate.
### Response: An API lets software talk to each other.""",
        "ground_truth": {
            "Retrieve": "[Retrieve:no]",
            "isREL": "[Relevant]",
            "isSUP": "[Partially supported]",
            "isUSE": "[Utility:3]"
        }
    },
    {
        "input": """### Instruction: What is cloud computing?
### Document: Cloud computing is the delivery of services over the internet.
### Response: Cloud computing allows users to access computing resources remotely.""",
        "ground_truth": {
            "Retrieve": "[Retrieve:yes]",
            "isREL": "[Relevant]",
            "isSUP": "[Fully supported]",
            "isUSE": "[Utility:5]"
        }
    }
]

# -------------------------------
# Inference
# -------------------------------
for i, item in enumerate(test_inputs, 1):
    sample_input = item["input"]
    expected = item.get("ground_truth", {})

    print(f"\n--- Test {i} ---")
    input_ids = encode(sample_input, vocab)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # Generate reflection with improved method
    with torch.no_grad():
        decoder_input = torch.tensor([[vocab["[PAD]"]]], dtype=torch.long)
        generated = model.generate(input_tensor, decoder_input, vocab=vocab, max_length=10)

    generated_ids = generated[0].tolist()
    tokens = decode_tokens(generated_ids, vocab)
    parsed = extract_reflection_fields(tokens)

    print(f"Input: {sample_input}")
    print(f"Generated tokens: {tokens}")
    print(f"Parsed reflection:")
    for key in ["Retrieve", "isREL", "isSUP", "isUSE"]:
        gt = expected.get(key, "[N/A]")
        print(f"  {key}: {parsed[key]} (expected: {gt})")

# -------------------------------
# Token Legend
# -------------------------------
print("\n" + "=" * 50)
print("Reflection Token Meanings:")
print("[retrieve:yes] - Should retrieve more information")
print("[retrieve:no] - No need for additional retrieval")
print("[relevant] - Document is relevant to the question")
print("[irrelevant] - Document is not relevant")
print("[fully supported] - Response is fully supported by document")
print("[partially supported] - Response is partially supported")
print("[no support] - Response has no support from document")
print("[Utility:1-5] - Response utility rating (1=low, 5=high)")
print("=" * 50)
