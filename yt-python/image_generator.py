import os
import re
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

STABILITY_API_KEY = os.getenv('STABILITY_API_KEY', 'sk-eR8zO8lXv8lglgjUz4O8ttX2yi9ftieJ9i2ZCheQd92KsGFS')
STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"

BASE_PROJECT_DIR = Path(__file__).parent.parent.resolve() 

MAX_WORKERS = 5
DEFAULT_STYLE_PRESET = "cinematic"
DEFAULT_SEED = 12345 
OUTPUT_FORMAT = "webp"

# --- SAFETY: TESTING MODE ---
# Set to True to generate only a few images (saves credits)
# Set to False to generate the whole story
TESTING_MODE_FLAG = False 
TEST_LIMIT = 2 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. DATA PARSING
# ==============================================================================

def parse_image_prompts(prompts_file_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    content = ""
    try:
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {prompts_file_path}")
        return []
        
    prompts_list = []
    blocks = re.split(r'(?=\={3}\s*SCENE\s+\d+.*?\={3})', content, flags=re.IGNORECASE | re.DOTALL)
    
    for i, block in enumerate(blocks):
        if not block.strip() or "SCENE" not in block: continue
            
        match = re.search(r'^\={3}\s*(.*?)\s*\={3}\s*(.*)', block.strip(), re.DOTALL)
        
        if match:
            full_header = match.group(1).strip()
            raw_prompt = match.group(2).strip()
            
            # --- MINIMAL CLEANING (Relying on Prompt Template for Syntax) ---
            # We only strip parameters that go into the API payload (--ar, --stylize)
            # We do NOT replace '::' or change text structure anymore.
            clean_prompt = re.sub(r'--ar\s+[\d:]+|--stylize\s+\d+|--v\s+\d+(\.\d+)?', '', raw_prompt)
            clean_prompt = " ".join(clean_prompt.split()).strip()

            # Naming Logic
            name_match = re.search(r'SCENE\s+(\d+):\s*(.*?)\s*-\s*SHOT\s+(\d+)\s*(.*)', full_header, re.IGNORECASE)
            scene_num = int(name_match.group(1)) if name_match and name_match.group(1).isdigit() else i
            shot_num = int(name_match.group(3)) if name_match and name_match.group(3).isdigit() else 1
            scene_name = name_match.group(2).strip().replace(':', '') if name_match else f"Scene{scene_num}"
            shot_type = name_match.group(4).strip().replace('=', '') if name_match else "Shot"
            
            ar_match = re.search(r'--ar\s+([\d:]+)', raw_prompt)
            aspect_ratio = ar_match.group(1) if ar_match else "16:9"

            prompts_list.append({
                'id': f"S{scene_num:02d}_SH{shot_num:02d}",
                'scene_name_short': f"{scene_name.split()[0]}{shot_type.split()[0]}",
                'prompt': clean_prompt,
                'aspect_ratio': aspect_ratio,
                'original_index': len(prompts_list) 
            })
            
        # Apply Testing Limit
        if limit is not None and len(prompts_list) >= limit:
            logger.info(f"🛑 Testing Limit Reached: Stopping parsing after {limit} prompts.")
            break
            
    return prompts_list

# ==============================================================================
# 3. API EXECUTION
# ==============================================================================

def generate_single_image(job_data: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
    # --- CONSOLE PREVIEW ---
    print(f"\n🎨 [PREVIEW] Generating {job_data['id']}:")
    print(f"   Prompt: {job_data['prompt'][:150]}...") 
    print(f"   Aspect Ratio: {job_data['aspect_ratio']}")
    
    payload = {
        "prompt": (None, job_data['prompt']),
        "output_format": (None, OUTPUT_FORMAT),
        "aspect_ratio": (None, job_data['aspect_ratio']),
        "style_preset": (None, DEFAULT_STYLE_PRESET),
        "seed": (None, str(DEFAULT_SEED))
    }
    headers = {
        "authorization": f"Bearer {api_key}",
        "accept": "image/*"
    }

    try:
        response = requests.post(STABILITY_API_URL, headers=headers, files=payload, timeout=120)
        if response.status_code == 200:
            job_data['image_bytes'] = response.content
            job_data['success'] = True
            logger.info(f"✨ Success: {job_data['id']}")
            return job_data
        else:
            logger.error(f"❌ API Error {job_data['id']}: {response.text[:200]}")
            job_data['success'] = False
            return job_data
    except Exception as e:
        logger.error(f"⛔ Network Error {job_data['id']}: {e}")
        job_data['success'] = False
        return job_data

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, default="channel", help="The channel folder name")
    parser.add_argument('--story', type=str, default="ranpuzel", help="The story folder name")
    args = parser.parse_args()

    # 1. Path Setup
    story_dir = BASE_PROJECT_DIR / args.channel / args.story
    prompts_file = story_dir / "image_prompt.txt"
    output_dir = story_dir / "images"
    
    if not output_dir.exists():
        logger.info(f"📁 Creating directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Determine Limit
    job_limit = TEST_LIMIT if TESTING_MODE_FLAG else None
    if TESTING_MODE_FLAG:
        logger.warning(f"🧪 TESTING MODE ACTIVE: Only generating first {TEST_LIMIT} images.")
    
    # 3. Parse with Limit
    jobs = parse_image_prompts(prompts_file, limit=job_limit)
    
    if not jobs:
        logger.warning("No jobs found.")
        return

    # 4. Execute
    logger.info(f"🚀 Starting generation for {len(jobs)} images...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(generate_single_image, job, STABILITY_API_KEY): job for job in jobs}
        
        results = []
        for future in as_completed(future_to_job):
            if future.result(): results.append(future.result())

    # 5. Save
    results.sort(key=lambda x: x['original_index'])
    
    for result in results:
        if not result.get('success'): continue
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', result['scene_name_short'])
        filename = f"{result['id']}_{sanitized_name}.{OUTPUT_FORMAT}"
        
        try:
            with open(output_dir / filename, 'wb') as f:
                f.write(result['image_bytes'])
        except Exception as e:
            logger.error(f"❌ Save Error: {e}")
            
    logger.info(f"✅ Pipeline Complete. Check {output_dir}")

if __name__ == "__main__":
    main()
