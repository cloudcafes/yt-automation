import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Load environment variables
load_dotenv()

# API Configuration
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', 'sk_24da67bfab1a2b87d79d4bad17d9c6e7fcc8dc9c3f04832a')

# Voice & Model Settings
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_flash_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"

BASE_PROJECT_DIR = Path(__file__).parent.parent.resolve() 

MAX_WORKERS = 5
DEFAULT_SEED = 12345 

# --- SAFETY: TESTING MODE ---
# Set to True to generate only a few lines (saves credits)
TESTING_MODE_FLAG = True 
TEST_LIMIT = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Client
try:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize ElevenLabs client: {e}")
    client = None

# ==============================================================================
# 2. DATA PARSING
# ==============================================================================

def parse_audio_segments(text_file_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    content = ""
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {text_file_path}")
        return []
        
    segments = []
    
    # 1. CLEANING THE TEXT
    # This regex removes ANYTHING inside square brackets.
    # It removes AND [brightly], [soft singing], etc.
    content = re.sub(r'\[.*?\]', '', content)
    
    # Split by lines
    lines = content.split('\n')
    
    valid_line_count = 0
    
    for i, line in enumerate(lines):
        clean_text = line.strip()
        
        # Skip empty lines
        if not clean_text: 
            continue
            
        # Skip file headers if present
        if "=====" in clean_text or "FILE:" in clean_text:
            continue

        valid_line_count += 1
        
        segments.append({
            'id': f"L{valid_line_count:03d}", # L001, L002...
            'text': clean_text,
            'original_index': valid_line_count
        })
            
        # Apply Testing Limit within the loop for efficiency
        if limit is not None and len(segments) >= limit:
            logger.info(f"🛑 Testing Limit Reached: Stopping parsing after {limit} segments.")
            break
            
    return segments

# ==============================================================================
# 3. API EXECUTION
# ==============================================================================

def generate_single_audio(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not client:
        return None

    # --- CONSOLE PREVIEW ---
    print(f"\n🎙️ [PREVIEW] Generating {job_data['id']}:")
    print(f"   Text: {job_data['text'][:100]}...") 
    
    try:
        # ElevenLabs returns a generator of bytes
        audio_stream = client.text_to_speech.convert(
            text=job_data['text'],
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT
        )
        
        # Consume the generator to get full bytes
        audio_bytes = b""
        for chunk in audio_stream:
            if chunk:
                audio_bytes += chunk

        if len(audio_bytes) > 0:
            job_data['audio_bytes'] = audio_bytes
            job_data['success'] = True
            logger.info(f"✨ Success: {job_data['id']}")
            return job_data
        else:
            logger.error(f"❌ API returned empty bytes for {job_data['id']}")
            job_data['success'] = False
            return job_data
            
    except Exception as e:
        logger.error(f"⛔ Network/API Error {job_data['id']}: {e}")
        job_data['success'] = False
        return job_data

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, default="channel", help="The channel folder name")
    parser.add_argument('--story', type=str, default="ranpuzel", help="The story folder name")
    parser.add_argument('--file', type=str, default="audio-text.txt", help="Input text filename")
    args = parser.parse_args()

    # 1. Path Setup
    story_dir = BASE_PROJECT_DIR / args.channel / args.story
    input_file = story_dir / args.file
    
    # Fallback logic
    if not input_file.exists():
        logger.warning(f"⚠️ {input_file.name} not found. Checking narration.txt...")
        fallback_file = story_dir / "narration.txt"
        if fallback_file.exists():
            input_file = fallback_file
            logger.info(f"📂 Using: {input_file}")
        else:
            logger.error("❌ No text file found for audio generation.")
            return

    output_dir = story_dir / "audio"
    
    if not output_dir.exists():
        logger.info(f"📁 Creating directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Determine Limit
    job_limit = TEST_LIMIT if TESTING_MODE_FLAG else None
    if TESTING_MODE_FLAG:
        logger.warning(f"🧪 TESTING MODE ACTIVE: Only generating first {TEST_LIMIT} lines.")
    
    # 3. Parse with Limit
    jobs = parse_audio_segments(input_file, limit=job_limit)
    
    if not jobs:
        logger.warning("No text segments found.")
        return

    # 4. Execute
    logger.info(f"🚀 Starting generation for {len(jobs)} audio segments...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(generate_single_audio, job): job for job in jobs}
        
        results = []
        for future in as_completed(future_to_job):
            if future.result(): results.append(future.result())

    # 5. Save
    results.sort(key=lambda x: x['original_index'])
    
    for result in results:
        if not result.get('success'): continue
        
        # Create filename: L001_TheFirstMove.mp3
        # Clean text for filename (take first 20 chars, alphanumeric only)
        snippet = re.sub(r'[^a-zA-Z0-9]', '', result['text'][:20])
        filename = f"{result['id']}_{snippet}.mp3"
        
        try:
            with open(output_dir / filename, 'wb') as f:
                f.write(result['audio_bytes'])
        except Exception as e:
            logger.error(f"❌ Save Error: {e}")
            
    logger.info(f"✅ Pipeline Complete. Check {output_dir}")

if __name__ == "__main__":
    main()
