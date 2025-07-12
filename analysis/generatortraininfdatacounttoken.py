import json
import re
import matplotlib.pyplot as plt
from collections import Counter

file_path = '../data/selfrag_generator_train_augmented.jsonl'

token_pattern = re.compile(r'\[(Retrieve:(yes|no))\]|\[(Relevant|Irrelevant)\]|\[(Fully supported|Partially supported|No support)\]|\[(Utility:[1-5])\]')

counts = Counter()

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        text = data.get('text', '')
        tokens = token_pattern.findall(text)
        # tokens is a list of tuples with multiple groups, flatten and filter
        flat_tokens = []
        for groups in tokens:
            for token in groups:
                if token != '':
                    flat_tokens.append(token)
        counts.update(flat_tokens)

# Plot counts for main token categories (you can customize)
labels = list(counts.keys())
values = [counts[label] for label in labels]

plt.figure(figsize=(10,5))
plt.bar(labels, values)
plt.title('Reflection Token Frequency in Generator Training Data')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
ç