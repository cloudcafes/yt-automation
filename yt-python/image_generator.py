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
OUTPUT_DIR = "generated_images"

BASE_PROJECT_DIR = Path(__file__).parent.parent.resolve() 

MAX_WORKERS = 5
TESTING_MODE_FLAG = False # Set to False to generate ALL images
DEFAULT_STYLE_PRESET = "cinematic"
DEFAULT_SEED = 12345 
OUTPUT_FORMAT = "webp"

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
            
            # --- FIX: CLEANING FOR STABILITY AI ---
            # 1. Remove Midjourney Parameters (--ar, --stylize)
            clean_prompt = re.sub(r'--ar\s+[\d:]+|--stylize\s+\d+', '', raw_prompt)
            # 2. Replace Midjourney Separators (::) with commas
            clean_prompt = clean_prompt.replace('::', ',')
            # 3. Clean up extra spaces/newlines
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
            
        if limit is not None and len(prompts_list) >= limit: break
            
    return prompts_list

# ==============================================================================
# 3. API EXECUTION
# ==============================================================================

def generate_single_image(job_data: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
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
            logger.info(f"✨ Generated: {job_data['id']}")
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
    # --- FIX: DYNAMIC STORY ARGUMENT ---
    parser.add_argument('--story', type=str, default="ranpuzel", help="Name of the story folder (e.g. ranpuzel)")
    args = parser.parse_args()

    # Dynamic Path Construction
    prompts_file = BASE_PROJECT_DIR / "channel" / args.story / "image_prompt.txt"
    output_dir = BASE_PROJECT_DIR / "channel" / args.story / "images"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📂 Reading from: {prompts_file}")
    
    jobs = parse_image_prompts(prompts_file, limit=None) # Set limit=2 for testing
    
    if not jobs:
        logger.warning("No jobs found. Check file path.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(generate_single_image, job, STABILITY_API_KEY): job for job in jobs}
        
        results = []
        for future in as_completed(future_to_job):
            if future.result(): results.append(future.result())

    results.sort(key=lambda x: x['original_index'])
    
    for result in results:
        if not result.get('success'): continue
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', result['scene_name_short'])
        filename = f"{result['id']}_{sanitized_name}.{OUTPUT_FORMAT}"
        with open(output_dir / filename, 'wb') as f:
            f.write(result['image_bytes'])
            
    logger.info(f"✅ Images saved to: {output_dir}")

if __name__ == "__main__":
    main()