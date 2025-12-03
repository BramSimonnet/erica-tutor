import requests

LM_STUDIO_URL = "http://host.docker.internal:1234/v1/chat/completions"

LM_STUDIO_MODEL = "qwen2_5-7b-instruct"

def call_llm(prompt: str) -> str:
    """
    Send a prompt to the local Qwen2.5-7B-Instruct model running in LM Studio
    and return the text of the first response.
    """
    body = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }

    try:
        resp = requests.post(LM_STUDIO_URL, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("LLM ERROR:", e)
        return "{}"
