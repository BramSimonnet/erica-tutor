import requests
import os

LMSTUDIO_API_BASE = "http://host.docker.internal:1234/v1"
MODEL_NAME = os.getenv("LMSTUDIO_MODEL", "qwen2.5")

def ask_llm(prompt: str) -> str:
    try:
        response = requests.post(
            f"{LMSTUDIO_API_BASE}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an AI tutor."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            },
            timeout=60
        )

        data = response.json()

        return (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "LLM returned no text.")
        )

    except Exception as e:
        return f"LLM error: {str(e)}"
