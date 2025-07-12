# Self-RAG Seminar Project

A comprehensive implementation of Self-RAG (Self-Reflective Retrieval-Augmented Generation) for educational purposes. This project demonstrates how to build a system that can retrieve relevant information, generate responses, and critique its own outputs using reflection tokens.

## Project Overview

Self-RAG is an advanced RAG system that incorporates self-reflection capabilities. The model learns to:
- Decide when to retrieve information (`[Retrieve:yes/no]`)
- Evaluate relevance of retrieved passages (`[Relevant/Irrelevant]`)
- Assess support level for claims (`[Fully supported/Partially supported/No support]`)
- Rate utility of responses (`[Utility:1-5]`)

## Project Structure

```
self-ragcode/
├── generator.py                 # Generator model training and inference
├── generator_inference.py       # Interactive inference with Ollama
├── critique.py                  # Critic model training (DistilBERT-based)
├── critique_infer.py           # Critic model inference
├── prepare_generator_data.py   # Data preparation for generator training
├── dataset_retriever.py        # Dataset loading utilities
├── retrain_critic.py           # Critic model retraining script
├── debug_generation.py         # Debug utilities for generation
├── test_improved_inference.py # Enhanced inference testing
├── testllama.py               # Llama model testing
├── data/                      # Training and evaluation datasets
├── model/                     # Saved model checkpoints
├── generator_finetuned/       # Fine-tuned generator models
├── analysis/                  # Analysis and evaluation scripts
│   ├── critiqueevaluation.py
│   ├── generatortraininfdatacounttoken.py
│   └── graph.ipynb
└── env/                       # Virtual environment
```

## Quick Start

### Prerequisites

1. **Python Environment**: Python 3.8+
2. **Ollama**: For local LLM inference
3. **PyTorch**: For model training and inference
4. **Transformers**: Hugging Face transformers library

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd self-ragcode

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install torch transformers scikit-learn tqdm
pip install ollama  # For local LLM inference
```

### Basic Usage

1. **Train the Critic Model**:
```bash
python critique.py
```

2. **Prepare Generator Training Data**:
```bash
python prepare_generator_data.py
```

3. **Train the Generator Model**:
```bash
python generator.py
```

4. **Run Interactive Inference**:
```bash
python generator_inference.py
```

## 🔧 Core Components

### 1. Critic Model (`critique.py`)

A DistilBERT-based model that learns to evaluate:
- **Retrieval decisions**: Whether to retrieve more information
- **Relevance**: Whether retrieved passages are relevant
- **Support**: How well passages support claims
- **Utility**: Overall quality of responses

**Key Features**:
- Multi-task learning with 4 classification heads
- Early stopping with patience
- Validation accuracy tracking
- Model checkpointing

### 2. Generator Model (`generator.py`)

A fine-tuned language model that:
- Learns to generate with reflection tokens
- Implements adaptive retrieval during generation
- Uses tree decoding for candidate evaluation
- Incorporates critic feedback

**Special Tokens**:
- `[Retrieve:yes/no]`: Retrieval decisions
- `[Relevant/Irrelevant]`: Relevance assessment
- `[Fully/Partially/No support]`: Support evaluation
- `[Utility:1-5]`: Quality rating
- `<p>...</p>`: Passage wrapping

### 3. Inference System (`generator_inference.py`)

Interactive inference using Ollama:
- Adaptive retrieval based on critic decisions
- Explicit passage quoting with `<p>` tags
- Reflection token parsing and validation
- Multi-segment generation with stopping criteria

## 📊 Data Format

### Critic Training Data (`data/selfrag_it_critic.json`)
```json
{
  "input": "Instruction text...",
  "Retrieve": "[Retrieve:yes]",
  "isREL": "[Relevant]",
  "isSUP": "[Fully supported]",
  "isUSE": "[Utility:5]"
}
```

### Generator Training Data (`data/selfrag_generator_train_augmented.jsonl`)
```json
{
  "text": "Instruction\n<p> Retrieved passage </p> [Relevant] [Fully supported]\n[Retrieve:yes] Response [Utility:5]"
}
```

## 🎮 Interactive Usage

### Running the Inference System

```bash
python generator_inference.py
```

**Example Interaction**:
```
Enter instruction: What is machine learning?
Generated answer: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. <p> Machine learning algorithms build mathematical models based on sample data to make predictions or decisions. </p> [Retrieve:yes] [Relevant] [Fully supported] [Utility:5]
```

### Key Features:
- **Adaptive Retrieval**: Automatically decides when to retrieve more information
- **Explicit Quoting**: Retrieved passages are clearly marked with `<p>` tags
- **Self-Reflection**: Each response includes reflection tokens
- **Quality Control**: Stops generation when utility is maximized

## 🔍 Analysis and Evaluation

### Analysis Scripts (`analysis/`)

- **`critiqueevaluation.py`**: Evaluate critic model performance
- **`generatortraininfdatacounttoken.py`**: Analyze training data statistics
- **`graph.ipynb`**: Visualization and analysis notebooks

### Evaluation Metrics

- **Critic Accuracy**: Per-task classification accuracy
- **Generation Quality**: Utility scores and reflection consistency
- **Retrieval Effectiveness**: Relevance and support assessment

## Advanced Usage

### Custom Corpus Integration

1. Add your corpus to `data/corpus.txt`
2. Update retrieval function in `generator_inference.py`
3. Retrain models with your domain-specific data

### Model Customization

```python
# Custom critic model
class CustomCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom architecture
        
# Custom generator model
def custom_generate_with_retrieval(model, prompt, retriever, critic):
    # Your custom generation logic
```

### Fine-tuning on Custom Data

1. Prepare data in the required format
2. Update training scripts with your data paths
3. Adjust hyperparameters for your domain

## Performance Optimization

### Training Tips

- **Batch Size**: Adjust based on available GPU memory
- **Learning Rate**: Start with 2e-5 for critic, 5e-5 for generator
- **Early Stopping**: Use patience=3 to prevent overfitting
- **Mixed Precision**: Enable fp16 for faster training

### Inference Optimization

- **Model Quantization**: Use quantized models for faster inference
- **Caching**: Cache retrieved passages for repeated queries
- **Parallel Processing**: Use multiple workers for batch inference

