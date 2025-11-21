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
RUN_NARRATION = True

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-df60b28326444de6859976f6e603fd9c')
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TEMPERATURE = 0.7

PROMPT_FILE = "step-1_narration_framework_prompt.txt"
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
    """Build cross-platform paths"""
    script_dir = Path(__file__).parent
    
    if script_dir.name == "yt-python":
        project_root = script_dir.parent
    else:
        project_root = Path.cwd()
        for parent in project_root.parents:
            if (parent / "channel").exists() and (parent / "yt-python").exists():
                project_root = parent
                break
    
    channel_dir = project_root / channel_folder
    story_dir = channel_dir / story_folder
    
    return {
        'project_root': project_root,
        'channel_dir': channel_dir,
        'story_dir': story_dir,
        'story_file': story_dir / "story.txt",
        'prompt_file': channel_dir / PROMPT_FILE,
        'output_file': story_dir / "narration_framework.txt",
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                
            logger.debug(f" Interaction logged to {self.history_file}")
            
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
            logger.info(" Full prompt and story logged to history (appended)")

            # 2. Get conversation history
            conversation_history = self._get_conversation_history_for_llm()
            logger.info(f" Loaded conversation history ({len(conversation_history)} chars)")

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

            logger.info(" Sending request to DeepSeek AI with history as attachment...")
            
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
            logger.info(f"粒 Cleaned {len(narration) - len(cleaned_narration)} special chars")

            # 6. Update the PENDING response with actual response
            self._update_pending_response(cleaned_narration)
            logger.info("✅ LLM response updated in history file")
            
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
            
        logger.info(f" Read {len(content)} chars from {file_path}")
        return content
        
    except UnicodeDecodeError:
        fallback_encodings = ['windows-1252', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().strip()
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
        logger.info(f" Wrote {len(content)} chars to {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error writing to {file_path}: {e}")
        return False

# ===== VALIDATION =====
def validate_paths(paths: Dict[str, Path]) -> bool:
    """Validate required paths and files"""
    if not paths['project_root'].exists():
        logger.error(f"❌ Project root missing: {paths['project_root']}")
        return False
    
    for file_key in ['prompt_file', 'story_file']:
        file_path = paths[file_key]
        if not file_path.exists():
            logger.error(f"❌ Required file missing: {file_path}")
            return False
    
    logger.info("✅ All paths validated")
    return True

# ===== MAIN NARRATION FUNCTION =====
def generate_story_narration(paths: Dict[str, Path]) -> bool:
    """Generate narration for a story"""
    prompt_content = read_file(paths['prompt_file'])
    story_content = read_file(paths['story_file'])
    
    if not prompt_content or not story_content:
        logger.error("❌ Failed to read prompt or story")
        return False

    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])
    if not narrator.is_available:
        logger.error("❌ DeepSeek client unavailable")
        return False

    logger.info(" Generating narration...")
    narration_result = narrator.generate_narration(prompt_content, story_content)
    if not narration_result:
        logger.error("❌ Failed to generate narration")
        return False

    success = write_file(paths['output_file'], narration_result)
    if success:
        logger.info(f" Narration saved to {paths['output_file']}")
        return True
    return False

# ===== MAIN EXECUTION =====
def main():
    if not RUN_NARRATION:
        logger.info("⏸️ Narration disabled")
        return 0

    try:
        paths = build_paths(BASE_PROJECT, CHANNEL_FOLDER, STORY_FOLDER)
        logger.info(" Starting narration generation...")
        
        for key, path in paths.items():
            if key != 'project_root':  # Don't log project root as it's too long
                logger.info(f"   {key}: {path.name}")

        if not validate_paths(paths):
            return 1

        success = generate_story_narration(paths)
        if success:
            logger.info("✅ Narration completed!")
            return 0
        else:
            logger.error("❌ Narration failed!")
            return 1
            
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)