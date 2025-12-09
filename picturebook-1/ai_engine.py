import os
import requests
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI
from elevenlabs.client import ElevenLabs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, json_mode: bool = False) -> Optional[str]:
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=4000, 
                temperature=temperature, response_format=response_format
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Text Gen Failed: {e}")
            return None

class ImageGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.stability.ai/v2beta/stable-image/generate/core"

    def generate(self, prompt: str, output_path: Path, negative_prompt: str = "", aspect_ratio: str = "16:9") -> bool:
        headers = {"authorization": f"Bearer {self.api_key}", "accept": "image/*"}
        # Updated Payload to include negative_prompt
        payload = {
            "prompt": prompt, 
            "negative_prompt": negative_prompt,
            "output_format": "webp", 
            "aspect_ratio": aspect_ratio, 
            "style_preset": "cinematic"
        }
        try:
            files = {k: (None, str(v)) for k, v in payload.items()}
            response = requests.post(self.api_url, headers=headers, files=files, timeout=60)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                logger.error(f"❌ Image API Error: {response.text}")
                return False
        except Exception as e:
            logger.error(f"⛔ Image Net Error: {e}")
            return False

class AudioGenerator:
    def __init__(self, api_key: str, voice_id: str = "JBFqnCBsd6RMkjVDRZzb"):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id

    def generate(self, text: str, output_path: Path) -> bool:
        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text, voice_id=self.voice_id, model_id="eleven_flash_v2_5", output_format="mp3_44100_128"
            )
            audio_bytes = b"".join([chunk for chunk in audio_stream if chunk])
            if audio_bytes:
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                return True
            return False
        except Exception as e:
            logger.error(f"⛔ Audio Error: {e}")
            return False