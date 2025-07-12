# SELF-RAG Prototype: Generator + Critic with Toy Dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np
from tqdm import tqdm
import json
import os
import re

# ---------- Vocabulary and Tokenizer ----------

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[Retrieve:yes]", "[Retrieve:no]",
    "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]", "[No support]",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"
]

def build_vocab(texts, min_freq=2):
    """Build vocabulary with minimum frequency threshold"""
    word_freq = {}
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Start with special tokens
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(vocab)
    
    # Add words that meet minimum frequency
    for word, freq in word_freq.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    
    return vocab

def encode(text, vocab, max_length=None):
    """Encode text with optional length limiting"""
    tokens = text.lower().split()
    if max_length:
        tokens = tokens[:max_length]
    return [vocab.get(tok, vocab["[UNK]"]) for tok in tokens]

def decode(ids, vocab):
    """Decode token IDs back to text"""
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = []
    for id_val in ids:
        if id_val == vocab["[PAD]"]:
            break
        tokens.append(id_to_token.get(id_val.item(), '[UNK]'))
    return ' '.join(tokens)

# ---------- Dataset ----------

class SelfRAGDataset(Dataset):
    def __init__(self, examples, vocab, max_length=50):
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, label_text = self.examples[idx]
        input_ids = encode(input_text, self.vocab, self.max_length)
        label_ids = encode(label_text, self.vocab, self.max_length)
        return {
            'input_ids': input_ids,
            'labels': label_ids
        }

def collate_fn(batch):
    # Find max lengths for padding
    max_input_len = max(len(item['input_ids']) for item in batch)
    max_label_len = max(len(item['labels']) for item in batch)
    max_len = max(max_input_len, max_label_len)
    
    # Pad sequences
    padded_inputs = []
    padded_labels = []
    
    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']
        
        # Pad input_ids to max_len
        padded_input = input_ids + [0] * (max_len - len(input_ids))
        padded_inputs.append(padded_input)
        
        # Pad labels to max_len
        padded_label = labels + [0] * (max_len - len(labels))
        padded_labels.append(padded_label)
    
    return {
        'input_ids': torch.tensor(padded_inputs, dtype=torch.long),
        'labels': torch.tensor(padded_labels, dtype=torch.long)
    }

# ---------- Generator ----------

class GeneratorModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.dropout(x)
        out, _ = self.rnn(x)
        logits = self.fc(out)
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=0.8, top_k=10):
        """Improved text generation with temperature and top-k sampling"""
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices.gather(1, next_token_idx)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we generate a PAD token or reach max length
                if next_token.item() == 0 or generated.size(1) >= max_length:
                    break
                    
            return generated

# ---------- Critic ----------

class CriticModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids):
        x = self.embed(input_ids)
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state
        pooled = lstm_out[:, -1, :]
        logits = self.fc(pooled)
        return logits

# ---------- Retrieval Simulation ----------

class SimpleRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(documents)
    
    def retrieve(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        similarities = (query_vector @ self.doc_vectors.T).toarray()[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

# ---------- Improved Toy Dataset ----------

def build_toy_data():
    """More diverse and realistic training data"""
    questions = [
        ("What is the capital of France?", "Paris is the capital of France. [Relevant] [Fully supported] [Utility:5]"),
        ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet. [Relevant] [Fully supported] [Utility:5]"),
        ("What is 2 + 2?", "2 + 2 equals 4. [No support] [Utility:5]"),
        ("Name a mammal that flies.", "Bats are mammals that can fly. [Relevant] [Fully supported] [Utility:5]"),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system. [Relevant] [Fully supported] [Utility:5]"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa. [Relevant] [Fully supported] [Utility:5]"),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy. [Relevant] [Fully supported] [Utility:5]"),
        ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second. [Relevant] [Fully supported] [Utility:5]"),
        ("Who discovered gravity?", "Isaac Newton is credited with discovering the law of gravity. [Relevant] [Fully supported] [Utility:5]"),
        ("What is DNA?", "DNA is the molecule that carries genetic information in living organisms. [Relevant] [Fully supported] [Utility:5]")
    ]
    return questions * 10  # 100 samples

def build_retrieval_docs():
    """More comprehensive retrieval documents"""
    return [
        "Paris is the capital of France and a major European city known for its culture and history.",
        "William Shakespeare was an English playwright who wrote many famous works including Hamlet.",
        "Bats are the only mammals capable of sustained flight, using their wings to navigate.",
        "Jupiter is the largest planet in our solar system, with a mass greater than all other planets combined.",
        "Leonardo da Vinci was an Italian Renaissance artist who painted the famous Mona Lisa.",
        "Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose.",
        "The speed of light in vacuum is approximately 299,792 kilometers per second, a fundamental constant in physics.",
        "Isaac Newton formulated the law of universal gravitation and made significant contributions to physics and mathematics.",
        "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development and functioning of living organisms.",
        "The Eiffel Tower is an iconic iron lattice tower located in Paris, France.",
        "Romeo and Juliet is another famous tragedy written by William Shakespeare.",
        "Mathematics is the study of numbers, quantities, shapes, and patterns.",
        "Birds are not mammals; they are a separate class of animals called Aves."
    ]

# ---------- Training Functions ----------

def train_generator(model, dataloader, optimizer, loss_fn, vocab_size, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Generator Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        logits = model(input_ids)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size_out = logits.shape
        logits_flat = logits.view(-1, vocab_size_out)
        labels_flat = labels.view(-1)
        
        loss = loss_fn(logits_flat, labels_flat)
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def train_critic(model, dataloader, optimizer, loss_fn, vocab_size, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Critic Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        logits = model(input_ids)
        
        # For simplicity, we'll use the first token of labels as the target class
        target_classes = labels[:, 0] % 5  # Simple mapping to 5 classes
        
        loss = loss_fn(logits, target_classes)
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

# ---------- Evaluation ----------

def evaluate_generator(model, dataloader, loss_fn, vocab_size):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            logits = model(input_ids)
            
            batch_size, seq_len, vocab_size_out = logits.shape
            logits_flat = logits.view(-1, vocab_size_out)
            labels_flat = labels.view(-1)
            
            loss = loss_fn(logits_flat, labels_flat)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def evaluate_critic(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            logits = model(input_ids)
            target_classes = labels[:, 0] % 5
            
            loss = loss_fn(logits, target_classes)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == target_classes).sum().item()
            total += target_classes.size(0)
            num_batches += 1
    
    accuracy = correct / total if total > 0 else 0
    return total_loss / num_batches, accuracy

# ---------- Self-RAG Pipeline ----------

def self_rag_pipeline(generator, critic, retriever, query, vocab, max_length=30):
    """Complete SELF-RAG pipeline: retrieve, generate, critique, improve"""
    
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=2)
    print(f"Query: {query}")
    print(f"Retrieved documents: {retrieved_docs}")
    
    # Step 2: Generate initial response
    input_text = query
    input_ids = torch.tensor([encode(input_text, vocab, max_length=20)], dtype=torch.long)
    
    generated_ids = generator.generate(input_ids, max_length=max_length, temperature=0.7, top_k=5)
    generated_text = decode(generated_ids[0], vocab)
    print(f"Generated response: {generated_text}")
    
    # Step 3: Critique the response
    generated_ids_for_critic = torch.tensor([encode(generated_text, vocab, max_length=30)], dtype=torch.long)
    critic_logits = critic(generated_ids_for_critic)
    critic_score = torch.softmax(critic_logits, dim=1).max().item()
    print(f"Critic score: {critic_score:.4f}")
    
    return {
        'query': query,
        'retrieved_docs': retrieved_docs,
        'generated_response': generated_text,
        'critic_score': critic_score
    }

# ---------- Model Persistence ----------

def save_models(generator, critic, vocab, path="./models"):
    os.makedirs(path, exist_ok=True)
    
    torch.save(generator.state_dict(), f"{path}/generator.pth")
    torch.save(critic.state_dict(), f"{path}/critic.pth")
    
    with open(f"{path}/vocab.json", 'w') as f:
        json.dump(vocab, f)
    
    print(f"Models saved to {path}")

def load_models(generator, critic, vocab, path="./models"):
    generator.load_state_dict(torch.load(f"{path}/generator.pth"))
    critic.load_state_dict(torch.load(f"{path}/critic.pth"))
    
    with open(f"{path}/vocab.json", 'r') as f:
        loaded_vocab = json.load(f)
    
    print(f"Models loaded from {path}")
    return loaded_vocab

# ---------- Main ----------

def main():
    print("=== SELF-RAG Prototype Training ===")
    
    # Build data
    data = build_toy_data()
    retrieval_docs = build_retrieval_docs()
    all_text = [x for pair in data for x in pair]
    vocab = build_vocab(all_text, min_freq=1)  # Lower threshold for better coverage
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_set = SelfRAGDataset(data, vocab, max_length=30)
    dataloader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Initialize models
    vocab_size = len(vocab)
    gen_model = GeneratorModel(vocab_size)
    critic_model = CriticModel(vocab_size)
    
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=5e-4, weight_decay=1e-5)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    gen_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["[PAD]"])
    critic_loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    gen_losses = []
    critic_losses = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train generator
        gen_loss = train_generator(gen_model, dataloader, gen_optimizer, gen_loss_fn, vocab_size, epoch)
        gen_losses.append(gen_loss)
        
        # Train critic
        critic_loss = train_critic(critic_model, dataloader, critic_optimizer, critic_loss_fn, vocab_size, epoch)
        critic_losses.append(critic_loss)
        
        print(f"Epoch {epoch+1}: Generator Loss: {gen_loss:.4f}, Critic Loss: {critic_loss:.4f}")
    
    # Evaluation
    print("\n=== Evaluation ===")
    eval_gen_loss = evaluate_generator(gen_model, dataloader, gen_loss_fn, vocab_size)
    eval_critic_loss, eval_critic_acc = evaluate_critic(critic_model, dataloader, critic_loss_fn)
    
    print(f"Generator Evaluation Loss: {eval_gen_loss:.4f}")
    print(f"Critic Evaluation Loss: {eval_critic_loss:.4f}")
    print(f"Critic Accuracy: {eval_critic_acc:.4f}")
    
    # Save models
    save_models(gen_model, critic_model, vocab)
    
    # Test SELF-RAG pipeline
    print("\n=== SELF-RAG Pipeline Test ===")
    retriever = SimpleRetriever(retrieval_docs)
    
    test_queries = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What animals can fly?",
        "What is the largest planet?"
    ]
    
    for query in test_queries:
        result = self_rag_pipeline(gen_model, critic_model, retriever, query, vocab)
        print("-" * 50)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()