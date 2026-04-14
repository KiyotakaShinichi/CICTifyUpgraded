# vision_reader.py
import os
import aiohttp, base64
from typing import Optional

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"


def _groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()

async def read_image_with_groq(image_bytes: bytes, question: str = "Read all text and describe this image.") -> Optional[str]:
    """Use Groq Vision to OCR and understand any image (e.g., floor plan, diagram)."""
    try:
        api_key = _groq_api_key()
        if not api_key:
            return None
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an advanced OCR and visual reasoning assistant. "
                               "Extract text, room labels, and layout descriptions from images and floor plans."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
                    ]
                }
            ],
            "max_tokens": 1000,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[Groq Vision] Bad status {resp.status}")
                    print(await resp.text())
                    return None
    except Exception as e:
        print(f"[Groq Vision Error] {e}")
        return None



