import requests

def ollama_query(prompt, model="llama3.2", max_tokens=100):
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    prompt = "Explain quantum computing simply."
    reply = ollama_query(prompt)
    print("Ollama says:", reply)
