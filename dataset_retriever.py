import json
import re

with open("data/selfrag_it_critic.json", "r", encoding="utf-8") as f:
    critique_data = json.load(f)

corpus_set = set()

for entry in critique_data:
    input_text = entry.get("input", "")
    # Use regex to extract the Document part robustly
    match = re.search(r"### Document:(.*?)### Response:", input_text, re.DOTALL)
    if match:
        document_text = match.group(1).strip()
        if document_text:
            corpus_set.add(document_text)
    else:
        # In case "### Response:" is missing or format differs, try alternate extraction
        # Example: extract everything after "### Document:" if no response marker
        doc_start = input_text.find("### Document:")
        if doc_start != -1:
            document_text = input_text[doc_start + len("### Document:"):].strip()
            if document_text:
                corpus_set.add(document_text)

corpus = list(corpus_set)

# Save corpus to file
with open("data/corpus.txt", "w", encoding="utf-8") as f_out:
    for doc in corpus:
        f_out.write(doc + "\n")

print(f"Extracted {len(corpus)} unique documents into corpus.txt")
