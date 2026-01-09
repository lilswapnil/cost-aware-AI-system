import os
import requests

OPENAI_API_KEY = os.getenv("OPENAPI_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def call_openai_chat(model: str, prompt: str, max_tokens: int = 120, temperature: float = 0.7):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()
