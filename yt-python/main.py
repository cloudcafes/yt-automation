import re
import os
import urllib3
import httpx
from openai import OpenAI
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import time
import chardet
import re
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== CONFIGURATION VARIABLES =====
BASE_PROJECT = "yt-automation"
CHANNEL_FOLDER = "channel"
STORY_FOLDER = "ranpuzel"

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-df60b28326444de6859976f6e603fd9c')
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TEMPERATURE = 0.7

# Prompt files
STEP_FILES = {
    1: {"prompt": "step-1_narration_framework_prompt.txt", "output": "narration_framework.txt", "desc": "Narration Framework"},
    2: {"prompt": "step-2_narration_prompt.txt", "output": "narration.txt", "desc": "Final Narration"},
    3: {"prompt": "step-3_charactersheet_prompt.txt", "output": "character_sheet.txt", "desc": "Character Sheet"},
    4: {"prompt": "step-4_scene_prompt.txt", "output": "scenes.txt", "desc": "Scene Breakdown"},
    5: {"prompt": "step-5_image_prompt.txt", "output": "image_prompt.txt", "desc": "Image Prompts"}
}

HISTORY_FILE = "ai_history.json"  # Changed to JSON

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===== TEXT CLEANING =====
def clean_text_preserve_punctuation(text: str) -> str:
    if not text: return text
    text = re.sub(r'[*#`~^_\\|@\[\]{}()<>]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# ===== PATH HANDLING =====
def build_paths(base_project: str, channel_folder: str, story_folder: str) -> Dict[str, Path]:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "yt-python" else Path.cwd()
    
    channel_dir = project_root / channel_folder
    story_dir = channel_dir / story_folder
    
    paths = {
        'project_root': project_root,
        'channel_dir': channel_dir,
        'story_dir': story_dir,
        'story_file': story_dir / "story.txt",
        'history_file': channel_dir / HISTORY_FILE
    }
    
    # Add step-specific paths
    for step, data in STEP_FILES.items():
        paths[f'step{step}_prompt'] = channel_dir / data['prompt']
        paths[f'step{step}_output'] = story_dir / data['output']
        
    return paths

# ===== DEEPSEEK CLIENT =====
class DeepSeekNarrator:
    def __init__(self, api_key: str, history_file: Path):
        self.api_key = api_key
        self.history_file = history_file
        self.client = None
        self.is_available = False
        self.lock = threading.Lock()  # For thread-safe file writing
        self._initialize_client()
        self._migrate_legacy_history()

    def _initialize_client(self):
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            http_client = httpx.Client(verify=False, timeout=60.0)
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url="https://api.deepseek.com", 
                http_client=http_client, 
                max_retries=2
            )
            # Simple connection test
            self.client.models.list()
            logger.info("✅ DeepSeek AI client initialized")
            self.is_available = True
        except Exception as e:
            logger.error(f"❌ DeepSeek client failed: {e}")
            self.is_available = False

    def _migrate_legacy_history(self):
        """Handle migration from old .txt history to .json"""
        txt_path = self.history_file.with_suffix('.txt')
        if txt_path.exists() and not self.history_file.exists():
            logger.warning("Found legacy history file. Archiving as .old and starting fresh JSON history.")
            txt_path.rename(txt_path.with_suffix('.txt.old'))

    def _load_history(self) -> List[Dict]:
        if not self.history_file.exists():
            return []
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.error("⚠️ History file corrupted, starting fresh.")
            return []

    def _save_interaction(self, prompt: str, story: str, response: str):
        """Thread-safe JSON append"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "folder": STORY_FOLDER,
            "prompt": prompt,
            "story_snippet": story[:200] + "..." if len(story) > 200 else story,
            "response": response
        }
        with self.lock:
            history = self._load_history()
            history.append(entry)
            # Keep file size manageable (last 50 entries)
            if len(history) > 50:
                history = history[-50:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

    def generate_narration(self, prompt: str, story_content: str, use_history: bool = True) -> Optional[str]:
        if not self.is_available: return None

        try:
            messages = []
            
            # Inject history only if requested (Optimizes tokens for Step 5)
            if use_history:
                history_data = self._load_history()
                # Get last 3 relevant exchanges
                context_str = "\n".join([
                    f"User: {h['prompt']}\nAI: {h['response']}" 
                    for h in history_data[-3:]
                ])
                if context_str:
                    messages.append({"role": "system", "content": f"Previous Context:\n{context_str}"})

            user_content = f"{prompt}\n\n=== STORY DATA ===\n{story_content}"
            messages.append({"role": "user", "content": user_content})

            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=DEEPSEEK_MAX_TOKENS,
                temperature=DEEPSEEK_TEMPERATURE
            )
            
            narration = clean_text_preserve_punctuation(response.choices[0].message.content.strip())
            
            # Log successful interaction
            self._save_interaction(prompt, story_content, narration)
            
            return narration
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return None

# ===== FILE OPERATIONS =====
def read_file(path: Path) -> Optional[str]:
    if not path.exists(): return None
    try:
        # Try simple utf-8 first
        with open(path, 'r', encoding='utf-8') as f: return f.read().strip()
    except UnicodeDecodeError:
        # Fallback using chardet
        try:
            raw = path.read_bytes()
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
            return raw.decode(enc).strip()
        except Exception:
            return None

def write_file(path: Path, content: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: f.write(content)
        return True
    except Exception as e:
        logger.error(f"❌ Write error {path.name}: {e}")
        return False

# ===== STEP LOGIC =====
def parse_scenes_and_shots(content: str) -> List[Dict]:
    """
    Parses the new hierarchical format:
    SCENE X -> Setting Visual Anchor -> SHOT 1, SHOT 2, etc.
    """
    structured_data = []
    
    # Split by "SCENE" headers (using robust regex for "## SCENE 1" or "SCENE 1")
    scene_blocks = re.split(r'(?=#{0,2}\s*SCENE\s+\d+)', content)
    
    for block in scene_blocks:
        if not block.strip(): continue
        
        # 1. Extract Scene Header
        header_match = re.search(r'(#{0,2}\s*SCENE\s+\d+.*)', block)
        if not header_match: continue
        scene_header = header_match.group(1).strip()
        
        # 2. Extract Visual Anchor (The constant background)
        anchor_match = re.search(r'Setting Visual Anchor:\s*(.*?)(?=\*\*SHOT|\n\n|SHOT)', block, re.DOTALL | re.IGNORECASE)
        visual_anchor = anchor_match.group(1).strip() if anchor_match else "A cinematic background."
        
        # 3. Extract Shots
        # Split the block by "SHOT X" markers
        shot_splits = re.split(r'(?=\*\*SHOT\s+\d+|\bSHOT\s+\d+)', block)
        
        for segment in shot_splits:
            # Check if this segment is actually a shot
            shot_header_match = re.search(r'(\*\*SHOT\s+\d+|SHOT\s+\d+).*', segment)
            if shot_header_match:
                shot_header = shot_header_match.group(0).strip().replace('*', '')
                
                # Create a composite input for the AI
                # We combine the Scene Context + Visual Anchor + Specific Shot Details
                composite_input = (
                    f"CONTEXT: {scene_header}\n"
                    f"VISUAL ANCHOR (BACKGROUND): {visual_anchor}\n"
                    f"SPECIFIC SHOT DETAILS:\n{segment.strip()}"
                )
                
                structured_data.append({
                    'id': f"{scene_header} - {shot_header}",
                    'full_text': composite_input
                })
                
    return structured_data

def process_step_5_parallel(paths: Dict[str, Path]) -> bool:
    """Updated to handle Shot-level generation"""
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available: return False
    
    prompt_template = read_file(paths['step5_prompt'])
    scenes_content = read_file(paths['step4_output'])
    
    # USE THE NEW PARSER
    shots_to_process = parse_scenes_and_shots(scenes_content)
    
    if not shots_to_process:
        logger.error("❌ No shots found to process")
        return False
        
    logger.info(f"🚀 Starting parallel generation for {len(shots_to_process)} shots...")
    
    results = {}
    
    def process_single_shot(shot_data):
        # We send the specific prompt template + the specific shot data
        prompt_input = f"{prompt_template}\n\n=== INPUT DATA ===\n{shot_data['full_text']}"
        result = narrator.generate_narration(prompt_template, prompt_input, use_history=False)
        return shot_data['id'], result

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_shot = {executor.submit(process_single_shot, s): s for s in shots_to_process}
        
        for future in as_completed(future_to_shot):
            shot_id, output = future.result()
            if output:
                results[shot_id] = output
                logger.info(f"✅ Finished {shot_id}")
            else:
                results[shot_id] = "[Generation Failed]"

    # Reconstruct Output
    final_output = "IMAGE PROMPTS:\n\n"
    # Sort keys to ensure Scene 1 Shot 1 comes before Scene 1 Shot 2
    for key in sorted(results.keys()): 
        final_output += f"=== {key} ===\n{results[key]}\n\n"
        
    return write_file(paths['step5_output'], final_output)

def run_standard_step(step_num: int, paths: Dict[str, Path]) -> bool:
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available: return False

    config = STEP_FILES[step_num]
    prompt = read_file(paths[f'step{step_num}_prompt'])
    
    # Gather inputs based on step logic
    inputs = ""
    if step_num == 1:
        inputs = read_file(paths['story_file'])
    elif step_num == 2:
        story = read_file(paths['story_file'])
        framework = read_file(paths['step1_output'])
        if not framework: return False
        inputs = f"STORY:\n{story}\n\nFRAMEWORK:\n{framework}"
    elif step_num == 4:
        # Step 4 needs BOTH Narration and Character Sheets
        narration = read_file(paths['step2_output'])
        chars = read_file(paths['step3_output'])
        inputs = f"STORY:\n{narration}\n\nCHARACTER DESIGNS (STRICT REFERENCE):\n{chars}"
    elif step_num in [3]:
        inputs = read_file(paths['step2_output']) # Needs narration
        
    if not prompt or not inputs:
        logger.error(f"❌ Missing inputs for Step {step_num}")
        return False

    result = narrator.generate_narration(prompt, inputs, use_history=True)
    if result:
        write_file(paths[f'step{step_num}_output'], result)
        logger.info(f"✅ Step {step_num} ({config['desc']}) Complete")
        return True
    return False

# ===== MAIN EXECUTION =====
def main():
    parser = argparse.ArgumentParser(description="YouTube Automation Pipeline")
    parser.add_argument('--step', type=int, nargs='+', help="Run specific steps (e.g., --step 1 2 5)")
    parser.add_argument('--all', action='store_true', help="Run all steps 1-5")
    args = parser.parse_args()

    paths = build_paths(BASE_PROJECT, CHANNEL_FOLDER, STORY_FOLDER)
    
    # Determine steps to run
    steps_to_run = []
    if args.all:
        steps_to_run = [1, 2, 3, 4, 5]
    elif args.step:
        steps_to_run = sorted(args.step)
    else:
        print("Usage: python main.py --all OR --step 1 2 5")
        return

    logger.info(f"📂 Project Root: {paths['project_root']}")
    logger.info(f"📝 Story Folder: {STORY_FOLDER}")

    for step in steps_to_run:
        if step not in STEP_FILES:
            logger.warning(f"⚠️ Skipping invalid step {step}")
            continue
            
        logger.info(f"\n--- Running Step {step}: {STEP_FILES[step]['desc']} ---")
        
        if step == 5:
            success = process_step_5_parallel(paths)
        else:
            success = run_standard_step(step, paths)
            
        if not success:
            logger.error(f"⛔ Pipeline stopped at Step {step}")
            break

if __name__ == "__main__":
    main()