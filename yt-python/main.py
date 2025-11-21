import os
import urllib3
import httpx
from openai import OpenAI
import datetime
from pathlib import Path
from typing import Optional, List, Dict
import logging
import time
import chardet
import re

# ===== CONFIGURATION VARIABLES =====
BASE_PROJECT = "yt-automation"
CHANNEL_FOLDER = "channel"
STORY_FOLDER = "ranpuzel"

# Step control - enable/disable each step independently
RUN_STEP1_NARRATION = False  # Already completed
RUN_STEP2_NARRATION = False  # Already completed  
RUN_STEP3_NARRATION = False  # Already completed
RUN_STEP4_NARRATION = False  # Already completed
RUN_STEP5_NARRATION = True   # Current step - Image prompts

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-df60b28326444de6859976f6e603fd9c')
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TEMPERATURE = 0.7

# Step-specific prompt files
STEP1_PROMPT = "step-1_narration_framework_prompt.txt"
STEP2_PROMPT = "step-2_narration_prompt.txt"
STEP3_PROMPT = "step-3_charactersheet_prompt.txt"
STEP4_PROMPT = "step-4_scene_prompt.txt"
STEP5_PROMPT = "step-5_image_prompt.txt"  # Current step

HISTORY_FILE = "ai_history.txt"

# Step configuration with proper input/output mapping
STEP_CONFIG = {
    1: {
        'enabled': RUN_STEP1_NARRATION,
        'prompt_file': 'step1_prompt_file',
        'input_files': ['story_file'],  # Requires story.txt
        'output_file': 'step1_output_file',  # Creates narration_framework.txt
        'description': 'Narration Framework'
    },
    2: {
        'enabled': RUN_STEP2_NARRATION,
        'prompt_file': 'step2_prompt_file',
        'input_files': ['story_file', 'narration_framework_file'],  # Requires story.txt + Step 1 output
        'output_file': 'step2_output_file',  # Creates narration.txt
        'description': 'Final Narration'
    },
    3: {
        'enabled': RUN_STEP3_NARRATION,
        'prompt_file': 'step3_prompt_file',
        'input_files': ['step2_output_file'],  # Requires narration.txt (Step 2 output)
        'output_file': 'step3_output_file',  # Creates character_sheet.txt
        'description': 'Character Sheet'
    },
    4: {
        'enabled': RUN_STEP4_NARRATION,
        'prompt_file': 'step4_prompt_file',
        'input_files': ['step2_output_file'],  # Requires narration.txt (Step 2 output)
        'output_file': 'step4_output_file',  # Creates scenes.txt
        'description': 'Scene Breakdown'
    },
    5: {
        'enabled': RUN_STEP5_NARRATION,
        'prompt_file': 'step5_prompt_file',
        'input_files': ['step4_output_file'],  # Requires scenes.txt (Step 4 output)
        'output_file': 'step5_output_file',  # Creates image_prompt.txt
        'description': 'Image Prompts',
        'processing_mode': 'scene_by_scene'  # Special processing mode
    }
}

# ===== TEXT CLEANING =====
def clean_text_preserve_punctuation(text: str) -> str:
    """Clean text while preserving essential punctuation for narration"""
    if not text:
        return text
    text = re.sub(r'[*#`~^_\\|@\[\]{}()<>]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# ===== PATH HANDLING =====
def build_paths(base_project: str, channel_folder: str, story_folder: str) -> Dict[str, Path]:
    """Build cross-platform paths with dependency awareness"""
    script_dir = Path(__file__).parent
    
    # Find project root (handles both yt-python and direct execution)
    if script_dir.name == "yt-python":
        project_root = script_dir.parent
    else:
        project_root = Path.cwd()
    
    channel_dir = project_root / channel_folder
    story_dir = channel_dir / story_folder
    
    return {
        'project_root': project_root,
        'channel_dir': channel_dir,
        'story_dir': story_dir,
        
        # Source files (independent)
        'story_file': story_dir / "story.txt",
        
        # Intermediate files (dependencies)
        'narration_framework_file': story_dir / "narration_framework.txt",  # Step 1 output
        
        # Prompt files
        'step1_prompt_file': channel_dir / STEP1_PROMPT,
        'step2_prompt_file': channel_dir / STEP2_PROMPT,
        'step3_prompt_file': channel_dir / STEP3_PROMPT,
        'step4_prompt_file': channel_dir / STEP4_PROMPT,
        'step5_prompt_file': channel_dir / STEP5_PROMPT,  # Current step
        
        # Output files
        'step1_output_file': story_dir / "narration_framework.txt",
        'step2_output_file': story_dir / "narration.txt",
        'step3_output_file': story_dir / "character_sheet.txt",
        'step4_output_file': story_dir / "scenes.txt",
        'step5_output_file': story_dir / "image_prompt.txt",  # Current step output
        
        # History
        'history_file': channel_dir / HISTORY_FILE
    }

# ===== ENCODING DETECTION =====
def detect_encoding(file_path: Path) -> str:
    """Detect file encoding automatically"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            if result['confidence'] < 0.7:
                if any(char in raw_data for char in [b'\x96', b'\x97', b'\x91', b'\x92']):
                    encoding = 'windows-1252'
            return encoding
    except Exception:
        return 'utf-8'

# ===== LOGGING SETUP =====
def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

setup_logging()
logger = logging.getLogger(__name__)

# ===== DEEPSEEK CLIENT =====
class DeepSeekNarrator:
    def __init__(self, api_key: str, history_file: Path):
        self.api_key = api_key
        self.history_file = history_file
        self.client = None
        self.is_available = False
        self._initialize_client()

    def _initialize_client(self):
        """Initialize DeepSeek API client with SSL disabled"""
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            http_client = httpx.Client(verify=False, timeout=60.0)
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                http_client=http_client,
                max_retries=2
            )
            
            # Test connection
            self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            logger.info("✅ DeepSeek AI client initialized (SSL disabled)")
            self.is_available = True
            
        except Exception as e:
            logger.error(f"❌ DeepSeek client failed: {e}")
            self.client = None
            self.is_available = False

    def _log_interaction(self, prompt: str, story_content: str, narration: str = None):
        """Log COMPLETE interaction to history file (APPEND only)"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TIMESTAMP: {datetime.datetime.now().isoformat()}\n")
                f.write(f"STORY_FOLDER: {STORY_FOLDER}\n")
                f.write(f"PROMPT:\n{prompt}\n")
                f.write(f"STORY_CONTENT:\n{story_content}\n")
                if narration:
                    f.write(f"LLM_RESPONSE:\n{narration}\n")
                else:
                    f.write(f"LLM_RESPONSE: PENDING...\n")
                f.write(f"{'='*80}\n")
                
            logger.debug("Interaction logged to history file")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to log interaction: {e}")

    def _update_pending_response(self, narration: str):
        """Update only the PENDING response in the history file without overwriting"""
        try:
            if not self.history_file.exists():
                return
            
            # Read entire file
            with open(self.history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the last "PENDING..." and replace only that part
            if "LLM_RESPONSE: PENDING..." in content:
                # Split by the separator to find entries
                separator = '=' * 80
                entries = content.split(separator)
                
                if len(entries) >= 2:
                    # Find the index of the last entry with PENDING
                    pending_index = -1
                    for i in range(len(entries)-1, -1, -1):
                        if "LLM_RESPONSE: PENDING..." in entries[i]:
                            pending_index = i
                            break
                    
                    if pending_index != -1:
                        # Replace PENDING with actual response in this entry
                        updated_entry = entries[pending_index].replace(
                            "LLM_RESPONSE: PENDING...", 
                            f"LLM_RESPONSE:\n{narration}"
                        )
                        
                        # Rebuild content with updated entry
                        entries[pending_index] = updated_entry
                        updated_content = separator.join(entries)
                        
                        # Write back updated content
                        with open(self.history_file, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                        
                        logger.debug("✅ Updated pending response in history file")
                        return
            
            logger.warning("No pending response found to update")
                
        except Exception as e:
            logger.error(f"⚠️ Failed to update pending response: {e}")

    def _get_conversation_history_for_llm(self) -> str:
        """Get conversation history for LLM context"""
        try:
            if not self.history_file.exists():
                return "No previous conversation history available."
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history_content = f.read()
            
            # Return recent history (excluding current pending request)
            entries = history_content.split('=' * 80)
            # Get entries that have complete responses (no PENDING)
            complete_entries = [entry for entry in entries if "LLM_RESPONSE:" in entry and "PENDING..." not in entry]
            recent_complete = complete_entries[-3:]  # Last 3 complete interactions
            
            return "\n".join(recent_complete) if recent_complete else "No complete conversation history available."
            
        except Exception as e:
            logger.error(f"⚠️ Failed to read conversation history: {e}")
            return "Error reading conversation history."

    def generate_narration(self, prompt: str, story_content: str) -> Optional[str]:
        """Generate narration for the provided story"""
        if not self.is_available:
            logger.error("DeepSeek client not available")
            return None

        try:
            # 1. Log FULL interaction with PENDING response (APPENDS)
            self._log_interaction(prompt, story_content)
            logger.info("Full prompt and story logged to history (appended)")

            # 2. Get conversation history
            conversation_history = self._get_conversation_history_for_llm()
            logger.info(f"Loaded conversation history ({len(conversation_history)} chars)")

            # 3. Build messages with history as clearly marked "attachment"
            messages = []
            
            # Build the complete message with history clearly marked as attachment
            user_content = f"""

{prompt}

=== CURRENT STORY: ===
{story_content}

=== AI LLM CONVERSATION HISTORY ===
{conversation_history}

"""

            messages.append({"role": "user", "content": user_content})

            logger.info("Sending request to DeepSeek AI with history as attachment...")
            
            # 4. Call API
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=DEEPSEEK_MAX_TOKENS,
                temperature=DEEPSEEK_TEMPERATURE,
                stream=False
            )
            
            narration = response.choices[0].message.content.strip()
            
            # 5. Clean the narration
            cleaned_narration = clean_text_preserve_punctuation(narration)
            logger.info(f"Cleaned {len(narration) - len(cleaned_narration)} special chars")

            # 6. Update the PENDING response with actual response
            self._update_pending_response(cleaned_narration)
            logger.info("LLM response updated in history file")
            
            logger.info(f"✅ Successfully generated narration ({len(cleaned_narration)} chars)")
            return cleaned_narration
            
        except Exception as e:
            logger.error(f"❌ Failed to generate narration: {e}")
            return None

# ===== FILE OPERATIONS =====
def read_file(file_path: Path) -> Optional[str]:
    """Read file content with encoding detection"""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read().strip()
            
        logger.info(f"Read {len(content)} chars from {file_path.name}")
        return content
        
    except UnicodeDecodeError:
        fallback_encodings = ['windows-1252', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    logger.info(f"Read {len(content)} chars from {file_path.name} (encoding: {encoding})")
                    return content
            except UnicodeDecodeError:
                continue
        logger.error(f"❌ All encoding attempts failed for {file_path}")
        return None
    except Exception as e:
        logger.error(f"❌ Error reading {file_path}: {e}")
        return None

def write_file(file_path: Path, content: str) -> bool:
    """Write content to file (overwrite or create)"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Wrote {len(content)} chars to {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Error writing to {file_path}: {e}")
        return False

# ===== SCENE PROCESSING FUNCTIONS =====
def parse_scenes_from_file(scenes_content: str) -> List[Dict[str, str]]:
    """Parse scenes.txt and extract individual scenes with robust pattern matching"""
    scenes = []
    
    if not scenes_content:
        return scenes
    
    # Multiple patterns to catch different scene formats
    patterns = [
        r'(SCENE\s+\d+.*?)(?=SCENE\s+\d+|$)',  # SCENE 1, SCENE 2, etc.
        r'(Scene\s+\d+.*?)(?=Scene\s+\d+|$)',  # Scene 1, Scene 2, etc.
        r'(## Scene \d+.*?)(?=## Scene \d+|$)', # Markdown style
        r'(Scene\s+\d+.*?)(?=\n\s*\n|$)'        # Scene followed by blank lines
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, scenes_content, re.DOTALL | re.IGNORECASE)
        scenes_found = False
        
        for match in matches:
            scene_text = match.group(1).strip()
            # Minimum scene length check to avoid false positives
            if scene_text and len(scene_text) > 20:
                # Extract header (first line) and content
                lines = scene_text.split('\n')
                header = lines[0].strip() if lines else "Unknown Scene"
                content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else scene_text
                
                scenes.append({
                    'header': header,
                    'content': content,
                    'full_text': scene_text
                })
                scenes_found = True
        
        # If we found scenes with this pattern, use them
        if scenes_found:
            logger.info(f"Found {len(scenes)} scenes using pattern: {pattern}")
            break
    
    # If no scenes found with patterns, try to split by major separators
    if not scenes and len(scenes_content) > 50:
        logger.warning("No scenes found with patterns, attempting fallback parsing")
        # Fallback: split by double newlines and look for scene indicators
        sections = re.split(r'\n\s*\n', scenes_content)
        scene_count = 0
        for section in sections:
            section = section.strip()
            if section and len(section) > 20:
                scene_count += 1
                scenes.append({
                    'header': f"SCENE {scene_count}",
                    'content': section,
                    'full_text': section
                })
    
    return scenes

# ===== STEP PROCESSING ENGINE =====
def validate_step_paths(step_num: int, paths: Dict[str, Path]) -> bool:
    """Validate all files required for a step exist"""
    config = STEP_CONFIG[step_num]
    required_files = {
        f'Step {step_num} Prompt': paths[config['prompt_file']]
    }
    
    # Add input files
    for i, input_key in enumerate(config['input_files']):
        required_files[f'Step {step_num} Input {i+1}'] = paths[input_key]
    
    all_valid = True
    for file_desc, file_path in required_files.items():
        if not file_path.exists():
            logger.error(f"❌ {file_desc} missing: {file_path.name}")
            all_valid = False
        else:
            logger.info(f"✅ {file_desc}: {file_path.name}")
    
    return all_valid

def validate_step_dependencies(step_num: int, paths: Dict[str, Path]) -> bool:
    """Validate that all required dependencies for a step are available"""
    config = STEP_CONFIG[step_num]
    
    # Check if this step depends on previous steps
    dependency_steps = []
    for input_key in config['input_files']:
        if input_key.startswith('step') and 'output' in input_key:
            # Extract step number from key like 'step2_output_file'
            dep_step = int(''.join(filter(str.isdigit, input_key)))
            if dep_step < step_num:
                dependency_steps.append(dep_step)
    
    # Verify all dependency steps were completed (their output files exist)
    for dep_step in dependency_steps:
        dep_config = STEP_CONFIG[dep_step]
        dep_output_file = paths[dep_config['output_file']]
        if not dep_output_file.exists():
            logger.error(f"❌ Step {step_num} requires Step {dep_step} output: {dep_output_file.name}")
            logger.error(f"   Please run Step {dep_step} first or ensure the file exists")
            return False
        else:
            logger.info(f"✅ Step {dep_step} dependency verified: {dep_output_file.name}")
    
    return True

def build_step_content(step_num: int, prompt: str, inputs: List[str]) -> str:
    """Build combined content for specific step requirements"""
    if step_num == 1:
        # Step 1: prompt + story
        return f"{prompt}\n\n{inputs[0]}"
    
    elif step_num == 2:
        # Step 2: prompt + story + framework
        return f"{prompt}\n\n=== CURRENT STORY: ===\n{inputs[0]}\n\n=== NARRATION FRAMEWORK: ===\n{inputs[1]}"
    
    elif step_num == 3:
        # Step 3: prompt + narration (Step 2 output)
        return f"{prompt}\n\n{inputs[0]}"
    
    elif step_num == 4:
        # Step 4: prompt + narration (Step 2 output)  
        return f"{prompt}\n\n{inputs[0]}"
    
    elif step_num == 5:
        # Step 5: prompt + scenes content (will be handled separately in scene processing)
        return f"{prompt}\n\n{inputs[0]}"
    
    else:
        # Default fallback
        return f"{prompt}\n\n{inputs[0] if inputs else ''}"

def process_step5_scene_by_scene(paths: Dict[str, Path]) -> bool:
    """Process Step 5 by generating image prompts for each scene individually"""
    
    # 1. Read required files
    step5_prompt = read_file(paths['step5_prompt_file'])
    scenes_content = read_file(paths['step4_output_file'])  # scenes.txt
    
    if not all([step5_prompt, scenes_content]):
        logger.error("❌ Missing required files for Step 5")
        return False

    # 2. Parse scenes from scenes.txt
    scenes = parse_scenes_from_file(scenes_content)
    if not scenes:
        logger.error("❌ No scenes found in scenes.txt")
        return False
    
    logger.info(f"️ Found {len(scenes)} scenes to process")
    
    # 3. Initialize narrator
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available:
        logger.error("❌ DeepSeek client unavailable")
        return False

    # 4. Process each scene individually
    all_image_prompts = []
    successful_scenes = 0
    
    for i, scene in enumerate(scenes, 1):
        logger.info(f" Processing {scene['header']} ({i}/{len(scenes)})")
        
        # Build scene-specific content
        scene_content = f"""
{step5_prompt}

{scene['full_text']}
"""
        
        # Generate image prompt for this scene
        image_prompt = narrator.generate_narration(step5_prompt, scene_content)
        
        if image_prompt:
            # Add scene identifier to the prompt
            scene_prompt = f"=== {scene['header']} ===\n{image_prompt}\n\n"
            all_image_prompts.append(scene_prompt)
            successful_scenes += 1
            logger.info(f"✅ Generated image prompt for {scene['header']}")
        else:
            logger.error(f"❌ Failed to generate image prompt for {scene['header']}")
            # Continue with other scenes even if one fails
            all_image_prompts.append(f"=== {scene['header']} ===\n[Failed to generate image prompt]\n\n")

        # Add delay between API calls to avoid rate limiting (except for the last scene)
        if i < len(scenes):
            logger.info("⏳ Waiting 2 seconds before next scene...")
            time.sleep(2)

    # 5. Combine all scene prompts and write to file
    if all_image_prompts:
        final_output = "IMAGE PROMPTS BY SCENE:\n\n" + "".join(all_image_prompts)
        success = write_file(paths['step5_output_file'], final_output)
        if success:
            logger.info(f"✅ All image prompts saved to {paths['step5_output_file'].name}")
            logger.info(f" Successfully processed {successful_scenes}/{len(scenes)} scenes")
            return True
    
    return False

def process_step(step_num: int, paths: Dict[str, Path]) -> bool:
    """Process a single step with dependency validation"""
    config = STEP_CONFIG[step_num]
    
    logger.info(f"PROCESSING STEP {step_num}: {config['description']}")
    logger.info("=" * 60)
    
    # Special handling for Step 5 (scene-by-scene processing)
    if step_num == 5 and config.get('processing_mode') == 'scene_by_scene':
        return process_step5_scene_by_scene(paths)
    
    # 1. Validate dependencies
    if not validate_step_dependencies(step_num, paths):
        return False
    
    # 2. Validate required files exist
    if not validate_step_paths(step_num, paths):
        return False
        
    # 3. Read all required content
    prompt_content = read_file(paths[config['prompt_file']])
    input_contents = []
    for input_key in config['input_files']:
        content = read_file(paths[input_key])
        if content is None:
            return False
        input_contents.append(content)
    
    # 4. Build step-specific content
    combined_content = build_step_content(step_num, prompt_content, input_contents)
    logger.info(f"Combined content prepared: {len(combined_content)} chars")
    
    # 5. Generate output using AI
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available:
        logger.error("❌ DeepSeek client unavailable")
        return False
        
    logger.info(f"Generating {config['description']}...")
    result = narrator.generate_narration(prompt_content, combined_content)
    
    if not result:
        logger.error(f"❌ Failed to generate {config['description']}")
        return False

    # 6. Save output
    success = write_file(paths[config['output_file']], result)
    if success:
        logger.info(f"✅ {config['description']} saved to {paths[config['output_file']].name}")
        return True
    
    return False

# ===== MAIN EXECUTION =====
def main():
    """Main execution with configuration-driven step processing"""
    paths = build_paths(BASE_PROJECT, CHANNEL_FOLDER, STORY_FOLDER)
    logger.info(" Starting YT Automation Pipeline...")
    
    # Log available files
    logger.info(" File Status:")
    for key, path in paths.items():
        if 'file' in key and path.exists():
            logger.info(f"   ✅ {key}: {path.name}")
        elif 'file' in key:
            logger.warning(f"   ❌ {key}: {path.name}")

    # Process all enabled steps using configuration
    for step_num in sorted(STEP_CONFIG.keys()):
        config = STEP_CONFIG[step_num]
        if config['enabled']:
            success = process_step(step_num, paths)
            if not success:
                logger.error(f"❌ Step {step_num} failed!")
                return 1
            logger.info(f"✅ Step {step_num} completed successfully!\n")
        else:
            logger.info(f"⏭️  Step {step_num} skipped: {config['description']}")
        
    logger.info(" All enabled steps completed!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)