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
# These will work on both Windows and Linux
BASE_PROJECT = "yt-automation"  # Root project folder
CHANNEL_FOLDER = "channel"      # Fixed channel folder
STORY_FOLDER = "ranpuzel"       # This changes per story
RUN_NARRATION = True

# API Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-df60b28326444de6859976f6e603fd9c')
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TEMPERATURE = 0.7

# File Names
PROMPT_FILE = "step-1_narration_prompt_1.txt"
HISTORY_FILE = "ai_history.txt"

# ===== TEXT CLEANING CONFIGURATION =====
# Characters to remove from AI output
SPECIAL_CHARS_TO_REMOVE = r'[*#`~^_\\|@]'  # Remove these special characters
# Characters to replace with spaces
SPECIAL_CHARS_TO_REPLACE = r'[\[\]{}()<>]'  # Replace these with spaces

# ===== TEXT CLEANING FUNCTIONS =====
def clean_ai_output(text: str) -> str:
    """
    Clean special characters from AI output while preserving readability
    """
    if not text:
        return text
    
    # Remove markdown code blocks and formatting
    text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
    text = re.sub(r'`[^`]*`', '', text)  # Remove inline code
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
    
    # Remove specific special characters
    text = re.sub(SPECIAL_CHARS_TO_REMOVE, '', text)
    
    # Replace other special characters with spaces
    text = re.sub(SPECIAL_CHARS_TO_REPLACE, ' ', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean up multiple newlines
    text = text.strip()
    
    return text

def clean_text_preserve_punctuation(text: str) -> str:
    """
    Clean text while preserving essential punctuation for narration
    """
    if not text:
        return text
    
    # Keep essential punctuation: . , ! ? : ; " ' - 
    # Remove other special characters
    text = re.sub(r'[*#`~^_\\|@\[\]{}()<>]', '', text)
    
    # Clean up whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

# ===== CROSS-PLATFORM PATH HANDLING =====
def build_paths(base_project: str, channel_folder: str, story_folder: str) -> Dict[str, Path]:
    """
    Build paths that work on both Windows and Linux with the exact structure:
    yt-automation/
    ├── channel/
    │   └── ranpuzel/
    └── yt-python/
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Navigate to project root (yt-automation)
    if script_dir.name == "yt-python":
        project_root = script_dir.parent
    else:
        # If running from different location, assume current directory is project root
        project_root = Path.cwd()
        # Try to find yt-automation structure
        if (project_root / "channel").exists() and (project_root / "yt-python").exists():
            pass  # We're already in yt-automation
        else:
            # Look for yt-automation in parent directories
            for parent in project_root.parents:
                if (parent / "channel").exists() and (parent / "yt-python").exists():
                    project_root = parent
                    break
    
    # Build the exact directory structure
    channel_dir = project_root / channel_folder
    story_dir = channel_dir / story_folder
    
    return {
        'project_root': project_root,
        'channel_dir': channel_dir,
        'story_dir': story_dir,
        'story_file': story_dir / "story.txt",
        'prompt_file': channel_dir / PROMPT_FILE,  # In channel folder
        'output_file': story_dir / "narration.txt",
        'history_file': channel_dir / HISTORY_FILE  # In channel folder
    }

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
            
            # Clean the narration output before using it
            cleaned_narration = clean_text_preserve_punctuation(narration)
            logger.info(f"粒 Cleaned {len(narration) - len(cleaned_narration)} special characters from narration")
            
            # Log this interaction (both to file and to memory for future context)
            self._log_interaction(prompt, story_content, cleaned_narration)
            
            # Add to conversation history for future context
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": cleaned_narration})
            
            # Keep history manageable (last 6 interactions)
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]
            
            logger.info(f"✅ Successfully generated and cleaned narration ({len(cleaned_narration)} characters)")
            return cleaned_narration
            
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

# ===== FILE OPERATIONS =====
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
    # Check that project structure exists
    if not paths['project_root'].exists():
        logger.error(f"❌ Project root does not exist: {paths['project_root']}")
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

    # Write cleaned narration to output file
    success = write_file(paths['output_file'], narration_result)
    if success:
        logger.info(f" Cleaned narration successfully saved to {paths['output_file']}")
        return True
    
    return False

# ===== MAIN EXECUTION =====
def main():
    """Main execution function"""
    if not RUN_NARRATION:
        logger.info("⏸️ Narration generation is disabled (RUN_NARRATION = False)")
        return 0

    try:
        # Build paths using the exact directory structure
        paths = build_paths(BASE_PROJECT, CHANNEL_FOLDER, STORY_FOLDER)
        
        logger.info(" Starting story narration generation...")
        logger.info(f" Project root: {paths['project_root']}")
        logger.info(f" Channel folder: {paths['channel_dir']}")
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