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

# ===== CONFIGURATION VARIABLES =====
BASE_PATH = r"C:\dev\Youtube\channel"
STORY_FOLDER = "ranpuzel"
RUN_NARRATION = True

# API Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-df60b28326444de6859976f6e603fd9c')
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TEMPERATURE = 0.7

# File Names
PROMPT_FILE = "step-1_narration_prompt_1.txt"
HISTORY_FILE = "ai_history.txt"

# ===== ENCODING DETECTION =====
def detect_encoding(file_path: Path) -> str:
    """Detect file encoding automatically"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            confidence = result['confidence']
            
            # Common encoding fallbacks for Windows
            if confidence < 0.7:
                if b'\x96' in raw_data or b'\x97' in raw_data or b'\x91' in raw_data or b'\x92' in raw_data:
                    encoding = 'windows-1252'
                else:
                    encoding = 'utf-8'
            
            logger.info(f" Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        logger.warning(f"⚠️ Could not detect encoding for {file_path}, defaulting to utf-8: {e}")
        return 'utf-8'

# ===== PATH HANDLING =====
def build_paths(base_path: str, story_folder: str) -> Dict[str, Path]:
    """Build cross-platform paths that work on both Windows and Linux"""
    base = Path(base_path)
    story_dir = base / story_folder
    
    return {
        'base': base,
        'story_dir': story_dir,
        'story_file': story_dir / "story.txt",
        'prompt_file': base / PROMPT_FILE,
        'output_file': story_dir / "narration.txt",
        'history_file': base / HISTORY_FILE
    }

# ===== LOGGING SETUP =====
def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
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
        
        # Load previous interactions to maintain context
        self.conversation_history = self._load_conversation_history()

    def _initialize_client(self):
        """Initialize DeepSeek API client with SSL disabled as required"""
        try:
            # Disable SSL certificate verification as required
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            http_client = httpx.Client(verify=False, timeout=60.0)
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                http_client=http_client,
                max_retries=2
            )
            
            # Simple connection test
            test_response = self.client.chat.completions.create(
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

    def _load_conversation_history(self) -> List[Dict[str, str]]:
        """Load previous interactions to maintain context with DeepSeek AI"""
        history = []
        try:
            if self.history_file.exists():
                # Use encoding detection for history file too
                encoding = detect_encoding(self.history_file)
                with open(self.history_file, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # Parse history entries (simplified parsing)
                entries = content.split('=' * 50)
                for entry in entries[-10:]:  # Last 10 interactions to avoid token limits
                    if "User:" in entry and "Assistant:" in entry:
                        user_part = entry.split("User:")[1].split("Assistant:")[0].strip()
                        assistant_part = entry.split("Assistant:")[1].strip()
                        
                        history.append({"role": "user", "content": user_part})
                        history.append({"role": "assistant", "content": assistant_part})
                
                logger.info(f" Loaded {len(history)//2} previous interactions from history")
        except Exception as e:
            logger.warning(f"Could not load conversation history: {e}")
        
        return history

    def generate_narration(self, prompt: str, story_content: str) -> Optional[str]:
        """Generate narration for the provided story"""
        if not self.is_available:
            logger.error("DeepSeek client not available")
            return None

        try:
            # Build messages with history + new request
            messages = []
            
            # Add conversation history to maintain context
            messages.extend(self.conversation_history)
            
            # Add current request with story as attachment
            user_content = f"{prompt}\n\nSTORY TO NARRATE:\n{story_content}"
            messages.append({"role": "user", "content": user_content})

            logger.info(" Sending narration request to DeepSeek AI...")
            
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=DEEPSEEK_MAX_TOKENS,
                temperature=DEEPSEEK_TEMPERATURE,
                stream=False
            )
            
            narration = response.choices[0].message.content.strip()
            
            # Log this interaction (both to file and to memory for future context)
            self._log_interaction(prompt, story_content, narration)
            
            # Add to conversation history for future context
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": narration})
            
            # Keep history manageable (last 6 interactions)
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]
            
            logger.info(f"✅ Successfully generated narration ({len(narration)} characters)")
            return narration
            
        except Exception as e:
            logger.error(f"❌ Failed to generate narration: {e}")
            return None

    def _log_interaction(self, prompt: str, story_content: str, narration: str):
        """Log all interactions to history file as required"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Story Folder: {STORY_FOLDER}\n")
                f.write(f"User: {prompt}\n")
                f.write(f"Story Content: {story_content[:500]}...\n")  # Preview only
                f.write(f"Assistant: {narration}\n")
                f.write(f"{'='*50}\n")
                
            logger.debug(f" Interaction logged to {self.history_file}")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to log interaction: {e}")

# ===== FILE OPERATIONS WITH ENCODING FIX =====
def read_file(file_path: Path) -> Optional[str]:
    """Read file content with automatic encoding detection"""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Detect encoding first
        encoding = detect_encoding(file_path)
        
        # Try reading with detected encoding
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read().strip()
            
        if not content:
            logger.warning(f"File is empty: {file_path}")
            
        logger.info(f" Read {len(content)} characters from {file_path} (encoding: {encoding})")
        return content
        
    except UnicodeDecodeError as e:
        # Fallback to common Windows encodings
        logger.warning(f"UTF-8 failed, trying fallback encodings for {file_path}")
        fallback_encodings = ['windows-1252', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                logger.info(f" Successfully read with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
                
        logger.error(f"❌ All encoding attempts failed for {file_path}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error reading file {file_path}: {e}")
        return None

def write_file(file_path: Path, content: str) -> bool:
    """Write content to file, overwriting if exists or creating new"""
    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f" Successfully wrote {len(content)} characters to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error writing to file {file_path}: {e}")
        return False

# ===== VALIDATION =====
def validate_paths(paths: Dict[str, Path]) -> bool:
    """Validate that all required paths and files exist"""
    # Check that base directory exists
    if not paths['base'].exists():
        logger.error(f"❌ Base path does not exist: {paths['base']}")
        return False
    
    # Check that required input files exist
    required_files = ['prompt_file', 'story_file']
    for file_key in required_files:
        file_path = paths[file_key]
        if not file_path.exists():
            logger.error(f"❌ Required file not found: {file_path}")
            return False
    
    logger.info("✅ All paths and files validated successfully")
    return True

# ===== MAIN NARRATION FUNCTION =====
def generate_story_narration(paths: Dict[str, Path]) -> bool:
    """Main function to generate narration for a story"""
    
    # Read prompt and story content
    prompt_content = read_file(paths['prompt_file'])
    story_content = read_file(paths['story_file'])
    
    if not prompt_content or not story_content:
        logger.error("❌ Failed to read prompt or story content")
        return False

    # Initialize DeepSeek narrator
    narrator = DeepSeekNarrator(DEEPSEEK_API_KEY, paths['history_file'])

    if not narrator.is_available:
        logger.error("❌ DeepSeek client not available")
        return False

    # Generate narration
    logger.info(" Generating narration...")
    narration_result = narrator.generate_narration(prompt_content, story_content)

    if not narration_result:
        logger.error("❌ Failed to generate narration")
        return False

    # Write narration to output file (overwrite or create new as required)
    success = write_file(paths['output_file'], narration_result)
    if success:
        logger.info(f" Narration successfully saved to {paths['output_file']}")
        return True
    
    return False

# ===== MAIN EXECUTION =====
def main():
    """Main execution function"""
    if not RUN_NARRATION:
        logger.info("⏸️ Narration generation is disabled (RUN_NARRATION = False)")
        return 0

    try:
        # Build paths using the fixed base + variable story folder structure
        paths = build_paths(BASE_PATH, STORY_FOLDER)
        
        logger.info(" Starting story narration generation...")
        logger.info(f" Base path: {paths['base']}")
        logger.info(f" Story folder: {paths['story_dir']}")
        logger.info(f" Story file: {paths['story_file']}")
        logger.info(f" Prompt file: {paths['prompt_file']}")
        logger.info(f" Output file: {paths['output_file']}")
        logger.info(f" History file: {paths['history_file']}")

        # Validate paths
        if not validate_paths(paths):
            return 1

        # Generate narration
        success = generate_story_narration(paths)
        
        if success:
            logger.info("✅ Story narration completed successfully!")
            return 0
        else:
            logger.error("❌ Story narration failed!")
            return 1
            
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)