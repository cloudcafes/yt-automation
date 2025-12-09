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
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys

# ===== CONFIGURATION VARIABLES =====
BASE_PROJECT = "yt-automation"
CHANNEL_FOLDER = "channel"
STORY_FOLDER = "ranpuzel"

# Ensure you have your API Key set in environment variables or hardcoded here safely
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-df60b28326444de6859976f6e603fd9c')
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TEMPERATURE = 0.7

# Prompt files mapping - UPDATED WITH STEP 7
STEP_FILES = {
    1: {"prompt": "step-1_narration_framework_prompt.txt", "output": "narration_framework.txt", "desc": "Narration Framework"},
    2: {"prompt": "step-2_narration_prompt.txt", "output": "narration.txt", "desc": "Final Narration"},
    3: {"prompt": "step-3_charactersheet_prompt.txt", "output": "character_sheet.txt", "desc": "Character Sheet"},
    4: {"prompt": "step-4_scene_prompt.txt", "output": "scenes.txt", "desc": "Scene Breakdown"},
    5: {"prompt": "step-5_image_prompt.txt", "output": "image_prompt.txt", "desc": "Image Prompts"},
    6: {"prompt": "step-6_video_metadata.txt", "output": "video_metadata.txt", "desc": "YouTube Metadata"},
    7: {"prompt": "step-7-narration-enhancement.txt", "output": "audio-text.txt", "desc": "Audio Enhanced Text"}
}

HISTORY_FILE = "ai_history.json"

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===== FILE CLEANUP FUNCTION =====
def cleanup_output_file(file_path: Path) -> bool:
    """
    Delete the output file before generating new content.
    Returns True if file was deleted or didn't exist, False on error.
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️  Deleted existing file: {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete {file_path.name}: {e}")
        return False

# ===== GIT AUTO-COMMIT FUNCTIONALITY =====
def git_auto_commit(story_folder: str, steps_run: List[int]) -> bool:
    """
    Automatically add and commit changes to git repository.
    Returns True if successful, False otherwise. Never raises exceptions.
    """
    try:
        # Get the project root directory (should be /root/Desktop/yt-automation)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent if script_dir.name == "yt-python" else Path.cwd()
        
        # Check if we're in a git repository
        if not (project_root / ".git").exists():
            logger.info("ℹ️ Not a git repository, skipping auto-commit")
            return False
        
        # Change to project root directory
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            # Check git status first
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if status_result.returncode != 0:
                logger.warning(f"⚠️ Git status check failed: {status_result.stderr[:100]}")
                return False
            
            # Check if there are any changes
            if not status_result.stdout.strip():
                logger.info("📭 No changes to commit")
                return True
            
            logger.info(f"📝 Found changes to commit:\n{status_result.stdout}")
            
            # Add all changes
            add_result = subprocess.run(
                ["git", "add", "."],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if add_result.returncode != 0:
                logger.warning(f"⚠️ Git add failed: {add_result.stderr[:100]}")
                return False
            
            logger.info("✅ Git add successful")
            
            # Create commit message
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            steps_str = ", ".join([f"Step {s}" for s in steps_run])
            commit_message = f"Auto-commit: {story_folder} - {steps_str} - {timestamp}"
            
            # Commit changes
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if commit_result.returncode != 0:
                # This might happen if there are no actual changes after add
                if "nothing to commit" in commit_result.stdout.lower():
                    logger.info("📭 Nothing to commit (no actual changes)")
                    return True
                logger.warning(f"⚠️ Git commit failed: {commit_result.stderr[:100]}")
                return False
            
            logger.info(f"✅ Git commit successful: {commit_message}")
            
            # Try to push (but don't fail if it doesn't work)
            push_result = subprocess.run(
                ["git", "push"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if push_result.returncode == 0:
                logger.info("✅ Git push successful")
                return True
            else:
                logger.warning(f"⚠️ Git push failed (may need manual push): {push_result.stderr[:100]}")
                # Still return True because commit was successful
                return True
                
        finally:
            # Always return to original directory
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        logger.warning("⏰ Git operation timed out")
        return False
    except FileNotFoundError:
        logger.warning("🔧 Git command not found (git not installed)")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Git auto-commit error (non-critical): {str(e)[:100]}")
        return False

# ===== TEXT CLEANING =====
def clean_text_preserve_punctuation(text: str) -> str:
    if not text: return text
    # Clean some markdown artifacts but keep structure
    text = re.sub(r'[*`~^_\\|@\[\]{}()<>]', '', text) 
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
        self.lock = threading.Lock()
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
            
            self._save_interaction(prompt, story_content, narration)
            return narration
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return None

# ===== FILE OPERATIONS =====
def read_file(path: Path) -> Optional[str]:
    if not path.exists(): return None
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read().strip()
    except UnicodeDecodeError:
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

# ===== ROBUST PARSING LOGIC (FIXED) =====
def parse_scenes_robust(content: str) -> List[Dict]:
    """
    Robustly parses scenes using a state machine approach.
    Handles variations like:
    - "Scene 1" vs "SCENE 1" vs "## Scene 1"
    - "Shot 1" vs "SHOT 1" vs "1. Shot"
    """
    structured_data = []
    
    # Normalize line endings and split
    lines = content.replace('\r\n', '\n').split('\n')
    
    current_scene_header = "Unknown Scene"
    current_visual_anchor = "A cinematic background."
    current_shot_buffer = []
    current_shot_header = ""
    
    # Regex patterns for flexibility
    scene_pattern = re.compile(r'^(?:#+\s*)?(?:SCENE|Scene)\s+(\d+)', re.IGNORECASE)
    anchor_pattern = re.compile(r'(?:Visual Anchor|Setting|Background):\s*(.*)', re.IGNORECASE)
    shot_pattern = re.compile(r'^(?:#+\s*)?(?:SHOT|Shot)\s+(\d+)', re.IGNORECASE)

    def save_current_shot():
        if current_shot_header and current_shot_buffer:
            # Flatten buffer to text
            shot_text = "\n".join(current_shot_buffer).strip()
            # Create composite input for Step 5
            composite_input = (
                f"CONTEXT: {current_scene_header}\n"
                f"VISUAL ANCHOR: {current_visual_anchor}\n"
                f"SPECIFIC SHOT DETAILS:\n{shot_text}"
            )
            structured_data.append({
                'id': f"{current_scene_header} - {current_shot_header}",
                'full_text': composite_input
            })

    for line in lines:
        line_clean = line.strip().replace('*', '') # Remove bolding for checking
        
        # 1. Detect Scene Header
        if scene_pattern.search(line_clean):
            save_current_shot() # Save previous shot if exists
            current_scene_header = line.strip().replace('*', '').replace('#', '')
            # Reset for new scene
            current_shot_buffer = []
            current_shot_header = ""
            current_visual_anchor = "A cinematic background." # Reset anchor
            continue
            
        # 2. Detect Visual Anchor
        anchor_match = anchor_pattern.search(line_clean)
        if anchor_match:
            current_visual_anchor = anchor_match.group(1).strip()
            continue
            
        # 3. Detect Shot Header
        shot_match = shot_pattern.search(line_clean)
        if shot_match:
            save_current_shot() # Save previous shot
            current_shot_header = line.strip().replace('*', '')
            current_shot_buffer = [] # Start new buffer
            continue
            
        # 4. Accumulate Content (if inside a shot)
        if current_shot_header:
            current_shot_buffer.append(line)

    # Save the very last shot
    save_current_shot()
    
    return structured_data

# ===== STEP 5 LOGIC (FIXED) =====
def process_step_5_parallel(paths: Dict[str, Path]) -> bool:
    """Updated to handle Shot-level generation with Character Consistency Injection"""
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available: return False
    
    # Clean up output file before generation
    output_path = paths['step5_output']
    if not cleanup_output_file(output_path):
        logger.warning(f"⚠️ Could not clean up {output_path.name}, continuing...")
    
    prompt_template = read_file(paths['step5_prompt'])
    scenes_content = read_file(paths['step4_output'])
    
    # CRITICAL FIX 1: Read Character Sheet for visual consistency
    character_sheet = read_file(paths['step3_output']) 
    if not character_sheet:
        logger.warning("⚠️ Character sheet not found. Visuals may be inconsistent.")
        character_sheet = "No character definitions provided."
    
    # CRITICAL FIX 2: Use Robust Parser
    shots_to_process = parse_scenes_robust(scenes_content)
    
    if not shots_to_process:
        logger.error("❌ No shots found to process. Check Step 4 output format.")
        return False
        
    logger.info(f"🚀 Starting parallel generation for {len(shots_to_process)} shots...")
    
    results = {}
    
    def process_single_shot(shot_data):
        # CRITICAL FIX 3: Inject Character Data into the prompt
        prompt_input = (
            f"{prompt_template}\n\n"
            f"=== REFERENCE: CHARACTER VISUALS (STRICT ADHERENCE) ===\n"
            f"{character_sheet}\n\n"
            f"=== INPUT DATA (SCENE & SHOT) ===\n"
            f"{shot_data['full_text']}"
        )
        # We assume the Prompt Template instructs the AI to use this data
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
        
    return write_file(output_path, final_output)

# ===== STEP 6 LOGIC =====
def process_step_6_metadata(paths: Dict[str, Path]) -> bool:
    """Process Step 6: Generate YouTube metadata from narration script"""
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available: return False
    
    # Clean up output file before generation
    output_path = paths['step6_output']
    if not cleanup_output_file(output_path):
        logger.warning(f"⚠️ Could not clean up {output_path.name}, continuing...")
    
    # Read the prompt template for Step 6
    prompt_template = read_file(paths['step6_prompt'])
    if not prompt_template:
        logger.error("❌ Step 6 prompt template not found")
        return False
    
    # Read the narration script (main input for Step 6)
    narration_script = read_file(paths['step2_output'])  # narration.txt
    if not narration_script:
        logger.error("❌ Narration script not found for Step 6")
        return False
    
    # Optionally read framework for audience context (not required but helpful)
    framework_content = read_file(paths['step1_output'])
    
    # Prepare input for AI
    if framework_content:
        inputs = f"=== NARRATION SCRIPT ===\n{narration_script}\n\n=== AUDIENCE FRAMEWORK (for context) ===\n{framework_content}"
    else:
        inputs = f"=== NARRATION SCRIPT ===\n{narration_script}"
    
    logger.info("🚀 Starting Step 6: YouTube Metadata Generation...")
    
    # Generate metadata using AI
    metadata = narrator.generate_narration(prompt_template, inputs, use_history=True)
    
    if not metadata:
        logger.error("❌ Failed to generate metadata")
        return False
    
    # Save the output
    success = write_file(output_path, metadata)
    if success:
        logger.info("✅ Step 6 (YouTube Metadata) Complete")
    return success

# ===== STEP 7 LOGIC =====
def process_step_7_audio_text(paths: Dict[str, Path]) -> bool:
    """Process Step 7: Generate audio-enhanced text from narration"""
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available: return False
    
    # Clean up output file before generation
    output_path = paths['step7_output']
    if not cleanup_output_file(output_path):
        logger.warning(f"⚠️ Could not clean up {output_path.name}, continuing...")
    
    # Read the prompt template for Step 7
    prompt_template = read_file(paths['step7_prompt'])
    if not prompt_template:
        logger.error("❌ Step 7 prompt template not found")
        return False
    
    # Read the narration script (main input for Step 7)
    narration_script = read_file(paths['step2_output'])  # narration.txt
    if not narration_script:
        logger.error("❌ Narration script not found for Step 7")
        return False
    
    # Prepare input for AI - using narration as the dialogue to enhance
    inputs = f"=== NARRATION TO ENHANCE WITH AUDIO TAGS ===\n{narration_script}"
    
    logger.info("🚀 Starting Step 7: Audio Text Enhancement...")
    
    # Generate enhanced audio text using AI
    enhanced_audio_text = narrator.generate_narration(prompt_template, inputs, use_history=True)
    
    if not enhanced_audio_text:
        logger.error("❌ Failed to generate enhanced audio text")
        return False
    
    # Save the output
    success = write_file(output_path, enhanced_audio_text)
    if success:
        logger.info("✅ Step 7 (Audio Enhanced Text) Complete")
    return success

# ===== STANDARD STEP PROCESSING =====
def run_standard_step(step_num: int, paths: Dict[str, Path]) -> bool:
    """Handle standard steps (1-4, 6, 7) that follow similar patterns"""
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available: return False

    config = STEP_FILES[step_num]
    prompt = read_file(paths[f'step{step_num}_prompt'])
    
    # Clean up output file before generation
    output_path = paths[f'step{step_num}_output']
    if not cleanup_output_file(output_path):
        logger.warning(f"⚠️ Could not clean up {output_path.name}, continuing...")
    
    # Gather inputs based on step logic
    inputs = ""
    if step_num == 1:
        inputs = read_file(paths['story_file'])
    elif step_num == 2:
        story = read_file(paths['story_file'])
        framework = read_file(paths['step1_output'])
        if not framework: return False
        inputs = f"STORY:\n{story}\n\nFRAMEWORK:\n{framework}"
    elif step_num == 3:
        inputs = read_file(paths['step2_output'])  # Needs narration
    elif step_num == 4:
        # Step 4 needs BOTH Narration and Character Sheets
        narration = read_file(paths['step2_output'])
        chars = read_file(paths['step3_output'])
        if not narration or not chars: return False
        inputs = f"STORY:\n{narration}\n\nCHARACTER DESIGNS (STRICT REFERENCE):\n{chars}"
    elif step_num == 6:
        # Handle Step 6 separately via process_step_6_metadata
        return process_step_6_metadata(paths)
    elif step_num == 7:
        # Handle Step 7 separately via process_step_7_audio_text
        return process_step_7_audio_text(paths)
        
    if not prompt or not inputs:
        logger.error(f"❌ Missing inputs for Step {step_num}")
        return False

    result = narrator.generate_narration(prompt, inputs, use_history=True)
    if result:
        write_file(output_path, result)
        logger.info(f"✅ Step {step_num} ({config['desc']}) Complete")
        return True
    return False

# ===== MAIN EXECUTION =====
def main():
    parser = argparse.ArgumentParser(description="YouTube Automation Pipeline")
    parser.add_argument('--step', type=int, nargs='+', help="Run specific steps (e.g., --step 1 2 5 6 7)")
    parser.add_argument('--all', action='store_true', help="Run all steps 1-7")
    parser.add_argument('--story', type=str, default=STORY_FOLDER, help=f"Story folder name (default: {STORY_FOLDER})")
    parser.add_argument('--no-git', action='store_true', help="Disable automatic git commit")
    args = parser.parse_args()

    # Use provided story folder or default
    story_folder = args.story
    paths = build_paths(BASE_PROJECT, CHANNEL_FOLDER, story_folder)
    
    # Validate story directory exists
    if not paths['story_dir'].exists():
        logger.error(f"❌ Story directory not found: {paths['story_dir']}")
        return
    
    # Determine steps to run
    steps_to_run = []
    if args.all:
        steps_to_run = [1, 2, 3, 4, 5, 6, 7]
    elif args.step:
        steps_to_run = sorted(set(args.step))  # Remove duplicates
    else:
        print("Usage: python main.py --all OR --step 1 2 5 6 7")
        print(f"Available steps: {list(STEP_FILES.keys())}")
        return

    logger.info(f"📂 Project Root: {paths['project_root']}")
    logger.info(f"📝 Story Folder: {story_folder}")

    # Track successful steps
    successful_steps = []
    
    for step in steps_to_run:
        if step not in STEP_FILES:
            logger.warning(f"⚠️ Skipping invalid step {step}")
            continue
            
        logger.info(f"\n--- Running Step {step}: {STEP_FILES[step]['desc']} ---")
        
        # Special handling for Step 5
        if step == 5:
            success = process_step_5_parallel(paths)
        else:
            success = run_standard_step(step, paths)
            
        if success:
            successful_steps.append(step)
        else:
            logger.error(f"⛔ Pipeline stopped at Step {step}")
            break
        
        # Brief pause between steps to avoid rate limiting
        time.sleep(1)

    # Summary of execution
    logger.info("\n" + "="*50)
    if successful_steps:
        logger.info(f"📊 Pipeline completed {len(successful_steps)}/{len(steps_to_run)} steps successfully")
        logger.info(f"✅ Successful steps: {', '.join([f'Step {s}' for s in successful_steps])}")
    else:
        logger.info("📊 No steps were completed successfully")
    
    # Auto git commit (if not disabled and we have successful steps)
    if successful_steps and not args.no_git:
        logger.info("\n📤 Attempting automatic git commit...")
        git_success = git_auto_commit(story_folder, successful_steps)
        if git_success:
            logger.info("✅ Git auto-commit completed")
        else:
            logger.info("ℹ️ Git auto-commit had issues (check logs above)")
    elif args.no_git:
        logger.info("🔒 Git auto-commit disabled (--no-git flag)")
    
    logger.info("\n🎬 Pipeline execution finished!")

if __name__ == "__main__":
    main()
