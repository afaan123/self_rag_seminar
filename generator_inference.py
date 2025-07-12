import subprocess
import json
import re

# === Load corpus ===
def load_corpus(path="data/corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

corpus = load_corpus()

# === Simple keyword overlap retriever ===
def retrieve_passages(query, top_k=3):
    query_words = set(query.lower().split())
    scored = []
    for doc in corpus:
        doc_words = set(doc.lower().split())
        score = len(query_words.intersection(doc_words))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored if score > 0][:top_k]
    if len(top_docs) < top_k:
        for doc in corpus:
            if doc not in top_docs:
                top_docs.append(doc)
            if len(top_docs) == top_k:
                break
    return top_docs

# === Dummy critic, replace with your real model inference ===
def critic_infer(instruction, passage, response_segment):
    return "[Relevant] [Fully supported]"

def critic_infer_use(instruction, response):
    return "[Utility:5]"

# === Parse reflection tokens from last line using regex ===
import re

def parse_reflection_tokens(text):
    # Flatten whitespace/newlines for easy matching
    flat_text = " ".join(text.strip().split())
    
    # Patterns for each token type, case-insensitive
    retrieve_pattern = r"\[Retrieve:(yes|no)\]"
    isrel_pattern = r"\[(Relevant|Irrelevant)\]"
    issup_pattern = r"\[(Fully supported|Partially supported|No support)\]"
    isuse_pattern = r"\[Utility:([1-5])\]"

    retrieve_match = re.search(retrieve_pattern, flat_text, re.IGNORECASE)
    isrel_match = re.search(isrel_pattern, flat_text, re.IGNORECASE)
    issup_match = re.search(issup_pattern, flat_text, re.IGNORECASE)
    isuse_match = re.search(isuse_pattern, flat_text, re.IGNORECASE)



    if retrieve_match and isrel_match and issup_match and isuse_match:
        return {
            "Retrieve": retrieve_match.group(0),
            "isREL": isrel_match.group(0),
            "isSUP": issup_match.group(0),
            "isUSE": isuse_match.group(0),
        }
    return None



# === Call Ollama CLI ===
def ollama_generate(prompt, model="llama3.2"):
    cmd = [
        "ollama",
        "run",
        model
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(input=prompt.encode())
    if proc.returncode != 0:
        print("Ollama error:", stderr.decode())
        return None
    return stdout.decode().strip()

# === Few-shot examples with retrieved passages quoted ===
FEW_SHOT = """
### Instruction: What is cloud computing?

Cloud computing is the delivery of computing services over the internet.

Retrieved passages:

<p> Cloud computing is the delivery of computing services over the internet. </p>

### Reflection tokens:
[Retrieve:yes] [Relevant] [Fully supported] [Utility:5]

---

### Instruction: What is a CPU?

CPU stands for Central Processing Unit, the brain of the computer that executes instructions.

Retrieved passages:

<p> CPU stands for Central Processing Unit, the primary component of a computer that performs most of the processing. </p>

### Reflection tokens:
[Retrieve:no] [Relevant] [Fully supported] [Utility:5]

---
"""

# === SELF-RAG adaptive generation loop with explicit quoting ===
def selfrag_generate(instruction, max_segments=3):
    generated_response = ""
    retrieve_decision = True
    final_reflection_tokens = None

    for segment_idx in range(max_segments):
        if retrieve_decision:
            retrieved = retrieve_passages(instruction, top_k=3)
            passages_with_reflections = []
            for passage in retrieved:
                reflection = critic_infer(instruction, passage, generated_response)
                # Keep passages clearly wrapped and reflection tokens appended
                passages_with_reflections.append(f"<p> {passage} </p> {reflection}")

            prompt = (
                FEW_SHOT
                + f"\n### Instruction: {instruction}\n"
                + "Please answer the question by quoting the retrieved passages exactly as they appear below.\n"
                + "Include the retrieved passages verbatim inside <p>...</p> tags in your answer.\n"
                + "\n".join(passages_with_reflections)
                + f"\n[Retrieve:yes] {generated_response}\n\n"
                + "### Reflection tokens:\n"
                + "Please output ONLY the reflection tokens exactly like:\n"
                + "[Retrieve:yes] [Relevant] [Fully supported] [Utility:5]\n"
            )
        else:
            prompt = (
                FEW_SHOT
                + f"\n### Instruction: {instruction}\n"
                + f"[Retrieve:no] {generated_response}\n\n"
                + "### Reflection tokens:\n"
                + "Please output ONLY the reflection tokens exactly like:\n"
                + "[Retrieve:no] [Relevant] [Fully supported] [Utility:5]\n"
            )

        output = ollama_generate(prompt)
        if output is None:
            print("Generation failed, stopping.")
            break

        # Split output to separate answer and reflection tokens block if present
        split_output = output.split("### Reflection tokens:")
        answer_text = split_output[0].strip()
        reflection_text = split_output[1].strip() if len(split_output) > 1 else ""
        
        # If no reflection tokens section found, try to extract tokens from the entire output
        if not reflection_text:
            # Look for reflection tokens in the entire output
            reflection_tokens = parse_reflection_tokens(output)
            if reflection_tokens:
                # Remove the reflection tokens from the answer text
                for token in [reflection_tokens["Retrieve"], reflection_tokens["isREL"], 
                            reflection_tokens["isSUP"], reflection_tokens["isUSE"]]:
                    answer_text = answer_text.replace(token, "").strip()
            else:
                print("Warning: Could not parse reflection tokens, stopping generation.")
                break
        else:
            reflection_tokens = parse_reflection_tokens(reflection_text)
            if reflection_tokens is None:
                print("Warning: Could not parse reflection tokens, stopping generation.")
                break

        generated_response += " " + answer_text
        final_reflection_tokens = reflection_tokens

        retrieve_decision = reflection_tokens["Retrieve"].lower() == "retrieve:yes"

        # Stop if utility is highest or last segment
        if reflection_tokens["isUSE"] == "[Utility:5]" or segment_idx == max_segments - 1:
            break

    # Add reflection tokens to the final output
    if final_reflection_tokens:
        reflection_string = f" {final_reflection_tokens['Retrieve']} {final_reflection_tokens['isREL']} {final_reflection_tokens['isSUP']} {final_reflection_tokens['isUSE']}"
        return (generated_response + reflection_string).strip()
    else:
        return generated_response.strip()

if __name__ == "__main__":
    print("SELF-RAG Ollama inference with explicit quoting. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter instruction: ")
        if user_input.lower() == "exit":
            break
        answer = selfrag_generate(user_input)
        print("\nGenerated answer:\n", answer)
