import os
import re
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ==============================================================================
# 1. CONFIGURATION AND CONSTANTS
# ==============================================================================

# --- Path and API Configuration (Adjust as needed) ---
# CRITICAL: Load API Key from environment variable or set here.
# NOTE: The provided key is a sample. Ensure it is replaced with a working key.
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY', 'sk-eR8zO8lXv8lglgjUz4O8ttX2yi9ftieJ9i2ZCheQd92KsGFS')

STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"
OUTPUT_DIR = "generated_images"

# --- Pipeline Configuration ---
BASE_PROJECT_DIR = Path(__file__).parent.parent.resolve() 
PROMPTS_FILE = BASE_PROJECT_DIR / "channel" / "ranpuzel" / "image_prompt.txt"

# --- Generation Parameters ---
MAX_WORKERS = 5
TESTING_MODE_FLAG = True 
TEST_LIMIT = 2
# Consistent Style: Using a style_preset saves prompt token space and guides the model.
DEFAULT_STYLE_PRESET = "cinematic"
# Using a fixed seed for testing helps confirm consistency. Use 0 for random.
DEFAULT_SEED = 12345 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 2. DATA PARSING AND PREPARATION
# ==============================================================================

def parse_image_prompts(prompts_file_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Reads image_prompt.txt and extracts structured prompt objects."""
    
    content = ""
    try:
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {prompts_file_path}")
        return []
        
    prompts_list = []
    
    # Regex to capture the header (Group 1) and the prompt content (Group 2)
    blocks = re.split(r'(?=\={3}\s*SCENE\s+\d+.*?\={3})', content, flags=re.IGNORECASE | re.DOTALL)
    
    for i, block in enumerate(blocks):
        if not block.strip() or "SCENE" not in block:
            continue
            
        match = re.search(r'^\={3}\s*(.*?)\s*\={3}\s*(.*)', block.strip(), re.DOTALL)
        
        if match:
            full_header = match.group(1).strip()
            prompt_content = match.group(2).strip()
            
            # Extract Scene and Shot numbers/names for file naming
            name_match = re.search(r'SCENE\s+(\d+):\s*(.*?)\s*-\s*SHOT\s+(\d+)\s*(.*)', full_header, re.IGNORECASE)
            
            # Fallback values for robustness
            scene_num = int(name_match.group(1)) if name_match and name_match.group(1).isdigit() else i
            shot_num = int(name_match.group(3)) if name_match and name_match.group(3).isdigit() else 1
            scene_name = name_match.group(2).strip().replace(':', '') if name_match else f"Scene{scene_num}"
            shot_type = name_match.group(4).strip().replace('=', '') if name_match else "Shot"
            
            # Extract aspect ratio from prompt footer (e.g., --ar 16:9)
            ar_match = re.search(r'--ar\s+([\d:]+)', prompt_content)
            aspect_ratio = ar_match.group(1) if ar_match else "16:9"

            # Clean up the prompt content by removing parameters like --ar and --stylize
            clean_prompt = re.sub(r'--ar\s+[\d:]+|--stylize\s+\d+', '', prompt_content).strip()

            prompts_list.append({
                'id': f"S{scene_num:02d}_SH{shot_num:02d}",
                'scene_name_short': f"{scene_name.split()[0]}{shot_type.split()[0]}",
                'full_header': full_header,
                'prompt': clean_prompt,
                'aspect_ratio': aspect_ratio,
                'original_index': len(prompts_list) 
            })
            
        if limit is not None and len(prompts_list) >= limit:
            break
            
    logger.info(f"✅ Parsed {len(prompts_list)} image generation jobs.")
    return prompts_list

# ==============================================================================
# 3. CONCURRENT API EXECUTION (FIXED)
# ==============================================================================

def generate_single_image(job_data: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
    """Handles the API call for one image."""
    
    output_format = "webp"
    
    # CRITICAL FIX: Preparing the payload for multipart/form-data
    # Use the tuple format (None, value) to ensure 'requests' sends form-data without a filename.
    payload = {
        "prompt": (None, job_data['prompt']),
        "output_format": (None, output_format),
        "aspect_ratio": (None, job_data['aspect_ratio']),
        "style_preset": (None, DEFAULT_STYLE_PRESET),
        "seed": (None, str(DEFAULT_SEED))
    }
    
    # Optional: Set client IDs for Stability AI tracking/support
    headers = {
        "authorization": f"Bearer {api_key}",
        "accept": "image/*", # Requesting raw image bytes for speed
        "stability-client-id": "yt-automation-pipeline",
        "stability-client-user-id": job_data['id']
    }

    try:
        response = requests.post(
            STABILITY_API_URL,
            headers=headers,
            # CRITICAL FIX: Use 'files' parameter for all data to force multipart/form-data encoding
            files=payload, 
            timeout=120
        )
        
        if response.status_code == 200:
            job_data['image_bytes'] = response.content
            job_data['output_format'] = output_format
            job_data['success'] = True
            logger.info(f"✨ Successfully generated: {job_data['id']}")
            return job_data
        
        else:
            # Handle non-200 errors (e.g., 400 Bad Request, 429 Rate Limit)
            try:
                error_details = response.json()
            except requests.exceptions.JSONDecodeError:
                error_details = response.text[:100] # Grab text if JSON fails
            
            logger.error(f"❌ API Failure for {job_data['id']} (Code: {response.status_code}): {error_details}")
            job_data['success'] = False
            return job_data

    except requests.exceptions.RequestException as e:
        logger.error(f"⛔ Network/Timeout Error for {job_data['id']}: {e}")
        job_data['success'] = False
        return job_data

# ==============================================================================
# 4. MAIN ORCHESTRATOR
# ==============================================================================

def main():
    # --- A. Setup ---
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    job_limit = TEST_LIMIT if TESTING_MODE_FLAG else None
    
    # --- B. Data Load ---
    jobs = parse_image_prompts(PROMPTS_FILE, limit=job_limit)
    if not jobs:
        logger.warning("Pipeline finished: No prompts to process.")
        return

    logger.info(f"🚀 Starting concurrent generation of {len(jobs)} images...")
    
    # --- C. Concurrent Execution ---
    results = []
    
    # Run the jobs concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {
            executor.submit(generate_single_image, job, STABILITY_API_KEY): job 
            for job in jobs
        }
        
        for future in as_completed(future_to_job):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Critical error during thread execution: {e}")

    # --- D. Sequential Saving and Naming ---
    
    # Sort results by the original sequence to ensure correct story order (S01_SH01, S01_SH02, etc.)
    results.sort(key=lambda x: x['original_index'])
    
    logger.info("💾 Starting sequential file saving...")
    
    successful_count = 0
    for result in results:
        if not result.get('image_bytes') or not result.get('success', False):
            continue
            
        # Create robust, sequentially numbered filename (e.g., S01_SH01_TheTowerEst.webp)
        # Sanitizes the name to prevent illegal characters in the filename
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', result['scene_name_short'])
        filename = (
            f"{result['id']}_"
            f"{sanitized_name}"
            f".{result['output_format']}"
        )
        
        output_path = Path(OUTPUT_DIR) / filename
        
        try:
            with open(output_path, 'wb') as f:
                f.write(result['image_bytes'])
            logger.info(f"✅ Saved: {filename}")
            successful_count += 1
            
        except IOError as e:
            logger.error(f"❌ Failed to save file {filename}: {e}")

    logger.info(f"\n--- Pipeline Summary ---")
    logger.info(f"Total jobs requested: {len(jobs)}")
    logger.info(f"Images successfully saved: {successful_count}")
    logger.info(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")
    

if __name__ == "__main__":
    main()