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
RUN_STEP3_NARRATION = True   # Current step - Character sheet
RUN_STEP4_NARRATION = False  # Scenes
RUN_STEP5_NARRATION = False  # Image prompts

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-df60b28326444de6859976f6e603fd9c')
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TEMPERATURE = 0.7

# Step-specific prompt files
STEP1_PROMPT = "step-1_narration_framework_prompt.txt"
STEP2_PROMPT = "step-2_narration_prompt.txt"
STEP3_PROMPT = "step-3_charactersheet_prompt.txt"  # Current step
STEP4_PROMPT = "step-4_scene_prompt.txt"
STEP5_PROMPT = "step-5_image_prompt.txt"

HISTORY_FILE = "ai_history.txt"

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
    """Build cross-platform paths for all steps"""
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
        
        # Input files
        'story_file': story_dir / "story.txt",
        'narration_framework_file': story_dir / "narration_framework.txt",
        
        # Prompt files for each step
        'step1_prompt_file': channel_dir / STEP1_PROMPT,
        'step2_prompt_file': channel_dir / STEP2_PROMPT,
        'step3_prompt_file': channel_dir / STEP3_PROMPT,  # Current step
        'step4_prompt_file': channel_dir / STEP4_PROMPT,
        'step5_prompt_file': channel_dir / STEP5_PROMPT,
        
        # Output files
        'step1_output_file': story_dir / "narration_framework.txt",
        'step2_output_file': story_dir / "narration.txt",
        'step3_output_file': story_dir / "character_sheet.txt",  # Current step output
        'step4_output_file': story_dir / "scenes.txt", 
        'step5_output_file': story_dir / "image_prompt.txt",
        
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

# ===== STEP 3 CHARACTER SHEET GENERATION =====
def validate_step3_paths(paths: Dict[str, Path]) -> bool:
    """Validate all files required for Step 3 exist"""
    required_files = {
        'Step 3 Prompt': paths['step3_prompt_file'],
        'Story File': paths['story_file']
    }
    
    all_valid = True
    for file_desc, file_path in required_files.items():
        if not file_path.exists():
            logger.error(f"❌ {file_desc} missing: {file_path}")
            all_valid = False
        else:
            logger.info(f"✅ {file_desc}: {file_path.name}")
    
    return all_valid

def generate_step3_character_sheet(paths: Dict[str, Path]) -> bool:
    """Generate character sheet using story + step3 prompt"""
    
    # Read all required inputs
    step3_prompt = read_file(paths['step3_prompt_file'])
    story_content = read_file(paths['story_file'])
    
    if not all([step3_prompt, story_content]):
        logger.error("❌ Missing required files for Step 3")
        return False

    # Build the combined content exactly as specified
    combined_content = f"""
{step3_prompt}

{story_content}
"""
    
    logger.info(f"Combined content prepared: {len(combined_content)} chars")
    
    # Initialize narrator
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available:
        logger.error("❌ DeepSeek client unavailable")
        return False

    # Generate character sheet
    logger.info("Generating character sheet (Step 3)...")
    character_sheet_result = narrator.generate_narration(step3_prompt, combined_content)
    
    if not character_sheet_result:
        logger.error("❌ Failed to generate character sheet")
        return False

    # Save to character_sheet.txt
    success = write_file(paths['step3_output_file'], character_sheet_result)
    if success:
        logger.info(f"Character sheet saved to {paths['step3_output_file'].name}")
        return True
    
    return False

# ===== STEP 2 NARRATION GENERATION =====
def validate_step2_paths(paths: Dict[str, Path]) -> bool:
    """Validate all files required for Step 2 exist"""
    required_files = {
        'Step 2 Prompt': paths['step2_prompt_file'],
        'Story File': paths['story_file'],
        'Narration Framework': paths['narration_framework_file']
    }
    
    all_valid = True
    for file_desc, file_path in required_files.items():
        if not file_path.exists():
            logger.error(f"❌ {file_desc} missing: {file_path}")
            all_valid = False
        else:
            logger.info(f"✅ {file_desc}: {file_path.name}")
    
    return all_valid

def generate_step2_narration(paths: Dict[str, Path]) -> bool:
    """Generate final narration using story + framework + step2 prompt"""
    
    # Read all required inputs
    step2_prompt = read_file(paths['step2_prompt_file'])
    story_content = read_file(paths['story_file']) 
    framework_content = read_file(paths['narration_framework_file'])
    
    if not all([step2_prompt, story_content, framework_content]):
        logger.error("❌ Missing required files for Step 2")
        return False

    # Build the combined content
    combined_content = f"""
{step2_prompt}

=== CURRENT STORY: ===
{story_content}

=== NARRATION FRAMEWORK: ===
{framework_content}
"""
    
    logger.info(f"Combined content prepared: {len(combined_content)} chars")
    
    # Initialize narrator
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available:
        logger.error("❌ DeepSeek client unavailable")
        return False

    # Generate narration
    logger.info("Generating final narration (Step 2)...")
    narration_result = narrator.generate_narration(step2_prompt, combined_content)
    
    if not narration_result:
        logger.error("❌ Failed to generate narration")
        return False

    # Save to narration.txt
    success = write_file(paths['step2_output_file'], narration_result)
    if success:
        logger.info(f"Final narration saved to {paths['step2_output_file'].name}")
        return True
    
    return False

# ===== STEP 1 NARRATION FRAMEWORK =====
def validate_step1_paths(paths: Dict[str, Path]) -> bool:
    """Validate all files required for Step 1 exist"""
    required_files = {
        'Step 1 Prompt': paths['step1_prompt_file'],
        'Story File': paths['story_file'],
    }
    
    all_valid = True
    for file_desc, file_path in required_files.items():
        if not file_path.exists():
            logger.error(f"❌ {file_desc} missing: {file_path}")
            all_valid = False
        else:
            logger.info(f"✅ {file_desc}: {file_path.name}")
    
    return all_valid

def generate_step1_narration(paths: Dict[str, Path]) -> bool:
    """Generate narration framework (Step 1)"""
    
    # Read all required inputs
    step1_prompt = read_file(paths['step1_prompt_file'])
    story_content = read_file(paths['story_file'])
    
    if not all([step1_prompt, story_content]):
        logger.error("❌ Missing required files for Step 1")
        return False

    # Initialize narrator
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available:
        logger.error("❌ DeepSeek client unavailable")
        return False

    # Generate narration framework
    logger.info("Generating narration framework (Step 1)...")
    framework_result = narrator.generate_narration(step1_prompt, story_content)
    
    if not framework_result:
        logger.error("❌ Failed to generate narration framework")
        return False

    # Save to narration_framework.txt
    success = write_file(paths['step1_output_file'], framework_result)
    if success:
        logger.info(f"Narration framework saved to {paths['step1_output_file'].name}")
        return True
    
    return False

# ===== MAIN EXECUTION =====
def main():
    """Main execution with step-based control"""
    paths = build_paths(BASE_PROJECT, CHANNEL_FOLDER, STORY_FOLDER)
    logger.info("Starting YT Automation Pipeline...")
    
    # Log available files
    logger.info("File Status:")
    for key, path in paths.items():
        if 'file' in key and path.exists():
            logger.info(f"   ✅ {key}: {path.name}")
        elif 'file' in key:
            logger.warning(f"   ❌ {key}: {path.name}")

    # Step 1: Narration Framework Generation
    if RUN_STEP1_NARRATION:
        logger.info("=" * 60)
        logger.info("PROCESSING STEP 1: Narration Framework")
        logger.info("=" * 60)
        
        if not validate_step1_paths(paths):
            logger.error("❌ Step 1 validation failed")
            return 1
            
        success = generate_step1_narration(paths)
        if not success:
            logger.error("❌ Step 1 failed!")
            return 1
        logger.info("✅ Step 1 completed successfully!")

    # Step 2: Final Narration Generation
    if RUN_STEP2_NARRATION:
        logger.info("=" * 60)
        logger.info("PROCESSING STEP 2: Final Narration")
        logger.info("=" * 60)
        
        if not validate_step2_paths(paths):
            logger.error("❌ Step 2 validation failed")
            return 1
            
        success = generate_step2_narration(paths)
        if not success:
            logger.error("❌ Step 2 failed!")
            return 1
        logger.info("✅ Step 2 completed successfully!")

    # Step 3: Character Sheet Generation
    if RUN_STEP3_NARRATION:
        logger.info("=" * 60)
        logger.info("PROCESSING STEP 3: Character Sheet")
        logger.info("=" * 60)
        
        if not validate_step3_paths(paths):
            logger.error("❌ Step 3 validation failed")
            return 1
            
        success = generate_step3_character_sheet(paths)
        if not success:
            logger.error("❌ Step 3 failed!")
            return 1
        logger.info("✅ Step 3 completed successfully!")

    # Future steps can be added here similarly
    if RUN_STEP4_NARRATION:
        logger.info("=" * 60)
        logger.info("PROCESSING STEP 4: Scenes")
        logger.info("=" * 60)
        logger.warning("⚠️ Step 4 not implemented yet")
        
    if RUN_STEP5_NARRATION:
        logger.info("=" * 60)
        logger.info("PROCESSING STEP 5: Image Prompts")
        logger.info("=" * 60)
        logger.warning("⚠️ Step 5 not implemented yet")
        
    logger.info("✅ All enabled steps completed!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)