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

load_dotenv()

# API Configuration
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', 'sk_24da67bfab1a2b87d79d4bad17d9c6e7fcc8dc9c3f04832a')

# Voice & Model Settings
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_flash_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"

BASE_PROJECT_DIR = Path(__file__).parent.parent.resolve() 
MAX_WORKERS = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Client
try:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize ElevenLabs client: {e}")
    client = None

# ==============================================================================
# 2. FILE CLEANUP FUNCTIONS
# ==============================================================================

def cleanup_output_directory(output_dir: Path) -> bool:
    """
    Clean up the output directory before generating new audio files.
    Removes all existing audio files in the directory.
    Returns True if successful, False on error.
    """
    try:
        if output_dir.exists():
            # Count existing files
            existing_files = list(output_dir.glob("*.mp3"))
            existing_count = len(existing_files)
            
            if existing_count > 0:
                logger.info(f"🗑️  Cleaning {existing_count} existing audio files...")
                
                # Delete all existing audio files
                for file_path in existing_files:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"⚠️ Could not delete {file_path.name}: {e}")
                
                logger.info(f"✅ Cleaned {existing_count} existing audio files")
            else:
                logger.info("📭 No existing audio files to clean")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clean output directory: {e}")
        return False

def cleanup_single_audio_file(output_dir: Path, file_pattern: str) -> bool:
    """
    Clean up a specific audio file before generating it.
    Returns True if file was deleted or didn't exist, False on error.
    """
    try:
        # Look for any file with this pattern
        existing_files = list(output_dir.glob(file_pattern))
        
        if existing_files:
            for file_path in existing_files:
                file_path.unlink()
                logger.debug(f"🗑️  Deleted existing: {file_path.name}")
        
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not clean up files for {file_pattern}: {e}")
        return False

# ==============================================================================
# 3. DATA PARSING
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
    
    # --- CLEANING STRATEGY ---
    
    # 1. Remove tags found in your data dump using Regex
    content = content.replace('\\', '')
    
    # 2. Remove backslashes safely using standard replace (NO REGEX HERE)
    content = content.replace('\\', '')
    
    # Split by lines
    lines = content.split('\n')
    
    valid_line_count = 0
    
    for i, line in enumerate(lines):
        clean_text = line.strip()
        
        # Skip empty lines
        if not clean_text: 
            continue
            
        # Skip file headers/separators if present in the data dump
        if "=====" in clean_text or "FILE:" in clean_text:
            continue

        valid_line_count += 1
        
        segments.append({
            'id': f"L{valid_line_count:03d}", # L001, L002...
            'text': clean_text,
            'original_index': valid_line_count,
            'line_number': valid_line_count
        })
            
        # Apply Testing Limit
        if limit is not None and len(segments) >= limit:
            logger.info(f"🛑 Testing Limit Reached: Stopping parsing after {limit} segments.")
            break
            
    return segments

# ==============================================================================
# 4. API EXECUTION
# ==============================================================================

def generate_single_audio(job_data: Dict[str, Any], output_dir: Path) -> Optional[Dict[str, Any]]:
    if not client:
        return None

    # Generate filename pattern for cleanup
    snippet = re.sub(r'[^a-zA-Z0-9]', '', job_data['text'][:20])
    filename_pattern = f"{job_data['id']}_*.mp3"
    
    # Clean up existing file for this job
    cleanup_single_audio_file(output_dir, filename_pattern)
    
    # --- CONSOLE PREVIEW ---
    # Show first 100 chars so you can verify if tags like [brightly] are present
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
# 5. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ElevenLabs Audio Generator")
    
    # REQUIRED Arguments (No defaults allowed to force explicit paths)
    parser.add_argument('--channel', type=str, required=True, help="The channel folder name (REQUIRED)")
    parser.add_argument('--story', type=str, required=True, help="The story folder name (REQUIRED)")
    
    # Optional Arguments
    parser.add_argument('--file', type=str, default="audio-text.txt", help="Input text filename (Default: audio-text.txt)")
    
    # Runtime Parameters (Testing, Limits, and Cleanup)
    parser.add_argument('--test', action='store_true', help="Enable testing mode (limits generation)")
    parser.add_argument('--limit', type=int, default=3, help="Number of lines to generate in test mode")
    parser.add_argument('--no-clean', action='store_true', help="Disable cleaning of output directory before generation")
    
    args = parser.parse_args()

    # 1. Path Setup
    story_dir = BASE_PROJECT_DIR / args.channel / args.story
    input_file = story_dir / args.file
    
    if not story_dir.exists():
        logger.error(f"❌ Story directory not found: {story_dir}")
        return

    # Fallback Logic
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
    elif not args.no_clean:
        # 2. Clean up output directory before generation
        logger.info("🧹 Cleaning output directory before generation...")
        cleanup_output_directory(output_dir)
    else:
        logger.info("🔒 Output directory cleaning disabled (--no-clean flag)")
    
    # 3. Determine Logic based on Runtime Flags
    if args.test:
        logger.warning(f"🧪 TESTING MODE ACTIVE: Generating only first {args.limit} lines.")
        job_limit = args.limit
    else:
        logger.info("🎬 FULL PRODUCTION MODE: Generating all lines.")
        job_limit = None
    
    # 4. Parse Jobs
    jobs = parse_audio_segments(input_file, limit=job_limit)
    
    if not jobs:
        logger.warning("No text segments found.")
        return

    # 5. Execute
    logger.info(f"🚀 Starting generation for {len(jobs)} audio segments...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(generate_single_audio, job, output_dir): job for job in jobs}
        
        results = []
        for future in as_completed(future_to_job):
            if future.result(): results.append(future.result())

    # 6. Save Results
    results.sort(key=lambda x: x['original_index'])
    
    for result in results:
        if not result.get('success'): continue
        
        # Create filename: L001_TheFirstMove.mp3
        # Use simple alphanumeric cleaning for filename, but keep original text for API
        snippet = re.sub(r'[^a-zA-Z0-9]', '', result['text'][:20])
        filename = f"{result['id']}_{snippet}.mp3"
        
        try:
            with open(output_dir / filename, 'wb') as f:
                f.write(result['audio_bytes'])
            logger.info(f"💾 Saved: {filename}")
        except Exception as e:
            logger.error(f"❌ Save Error: {e}")
    
    # 7. Summary
    success_count = sum(1 for r in results if r.get('success'))
    fail_count = len(results) - success_count
    
    logger.info(f"✅ Pipeline Complete. {success_count}/{len(results)} audio segments generated successfully")
    if fail_count > 0:
        logger.warning(f"⚠️ {fail_count} audio segments failed to generate")
    logger.info(f"📁 Check {output_dir}")

if __name__ == "__main__":
    main()
