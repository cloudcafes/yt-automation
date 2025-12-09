import os
import json
import re
import logging
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ai_engine import TextGenerator, ImageGenerator, AudioGenerator

# ==============================================================================
# 1. GLOBAL SETTINGS & TUNING
# ==============================================================================

PIPELINE_SETTINGS = {
    "PAGE_COUNT": 12,
    "IMAGE_ASPECT_RATIO": "16:9",  # Optimized for YouTube
    "AUDIO_VOICE_ID": "JBFqnCBsd6RMkjVDRZzb", # ElevenLabs Voice ID
    "MAX_WORKERS": 4, # Parallel threads for rendering
    "LLM_MODEL": "deepseek-chat" # Ensure you use a capable model
}

# API KEYS
API_KEYS = {
    "DEEPSEEK": os.getenv("DEEPSEEK_API_KEY", "sk-df60b28326444de6859976f6e603fd9c"),
    "STABILITY": os.getenv("STABILITY_API_KEY", "sk-eR8zO8lXv8lglgjUz4O8ttX2yi9ftieJ9i2ZCheQd92KsGFS"),
    "ELEVENLABS": os.getenv("ELEVENLABS_API_KEY", "sk_24da67bfab1a2b87d79d4bad17d9c6e7fcc8dc9c3f04832a")
}

# SYSTEM PROMPTS (OPTIMIZED)
PROMPTS = {
    "STAGE_0_ANALYSIS": """
        You are a professional Story Analyst. Extract constraints from the user request.
        Output purely JSON:
        {{
            "target_age": int,
            "narrative_goal": "string (The central conflict or lesson, e.g., 'Overcoming fear of the dark')",
            "tone": "string (e.g., Warm, Adventurous)",
            "format": "Picturebook"
        }}
    """,

    "STAGE_1_BLUEPRINT": """
        You are a Children's Book Architect. Create a creative brief.
        
        CRITICAL TASK: Define reusable, consistent prompt fragments for AI image generation.
        1. 'visual_style': Be highly specific (e.g., '3D Pixar style render, soft lighting, vibrant colors').
        2. 'consistent_char_prompt': Describe the protagonist fully as a reusable fragment (e.g., 'A 7-year-old boy with messy red hair and a green hoodie').
        3. 'negative_prompt': A list of things to avoid in the image.
        
        Output purely JSON:
        {{
            "title": "string",
            "protagonist": {{ "name": "string", "consistent_char_prompt": "string" }},
            "setting": "string",
            "visual_style": "string",
            "plot_summary": "string",
            "negative_prompt": "string (e.g., 'low quality, blurry, warped, extra fingers, text, watermark, bad anatomy')"
        }}
    """,

    "STAGE_2_SCRIPT": """
        You are a Storybook Scripter. Write exactly {page_count} pages.
        Each page text must be short (max 40 words), engaging, and suitable for TTS.
        
        CRITICAL TASK: For each page, include a 'visual_scene' key. This must be an objective description of the specific action, setting, and camera angle (e.g., 'wide shot', 'close-up') to be captured for the image.

        Output purely JSON List:
        [
            {{ 
                "page": 1, 
                "text": "The spoken words for the page.", 
                "emotional_beat": "Example: Nervous",
                "visual_scene": "A medium shot of Rory clutching a backpack on the doorstep of a cozy wooden house."
            }},
            ...
        ]
    """,

    "STAGE_3_VISUALS": """
        You are an AI Art Director. Your goal is MAXIMAL VISUAL CONSISTENCY.
        You must generate a prompt for each page using a strict formula.
        
        STRICT FORMULA:
        [Visual Style Anchor], [Visual Scene Content], [Protagonist Anchor], [Lighting & Mood based on Emotional Beat], high detail.
        
        MANDATORY INPUTS (Insert these exactly):
        STYLE_ANCHOR = "{style}"
        CHARACTER_ANCHOR = "{char_desc}"
        
        Input Data: {script_json}

        Output purely JSON List:
        [
            {{ "page": 1, "image_prompt": "{style}, [INSERT VISUAL SCENE HERE], {char_desc}, [lighting/mood], high detail." }},
            ...
        ]
    """
}

# ==============================================================================
# 2. THE PIPELINE CLASS
# ==============================================================================

class StorybookGenerator:
    def __init__(self, output_dir: str):
        self.base_dir = Path(output_dir)
        self.img_dir = self.base_dir / "images"
        self.audio_dir = self.base_dir / "audio"
        
        # Initialize AI Engines
        self.llm = TextGenerator(api_key=API_KEYS["DEEPSEEK"], model=PIPELINE_SETTINGS["LLM_MODEL"])
        self.painter = ImageGenerator(api_key=API_KEYS["STABILITY"])
        self.narrator = AudioGenerator(api_key=API_KEYS["ELEVENLABS"], voice_id=PIPELINE_SETTINGS["AUDIO_VOICE_ID"])
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _cleanup(self):
        """Wipes the output directory to ensure a fresh run."""
        if self.base_dir.exists():
            self.logger.warning(f"🧹 Cleaning up existing directory: {self.base_dir}")
            #shutil.rmtree(self.base_dir)
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)

    def _clean_json(self, text: str) -> dict:
        """Sanitizes LLM output to extract valid JSON."""
        try:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*$', '', text)
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match:
                text = match.group(0)
            return json.loads(text)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON: {text}")
            raise ValueError("LLM did not return valid JSON")

    def _save_file(self, filename: str, content: str):
        with open(self.base_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)

    def _save_json(self, filename: str, data):
        with open(self.base_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # --- STAGE 0: ANALYSIS ---
    def stage_0_analyze(self, user_prompt: str) -> dict:
        self.logger.info("🕵️ Stage 0: Analyzing Constraints...")
        response = self.llm.generate(PROMPTS["STAGE_0_ANALYSIS"], user_prompt, temperature=0.0, json_mode=True)
        data = self._clean_json(response)
        self._save_json("00_constraints.json", data)
        return data

    # --- STAGE 1: BLUEPRINT ---
    def stage_1_blueprint(self, constraints: dict) -> dict:
        self.logger.info("🏗️ Stage 1: Building Narrative World...")
        user_input = f"Create a blueprint based on these constraints: {json.dumps(constraints)}"
        response = self.llm.generate(PROMPTS["STAGE_1_BLUEPRINT"], user_input, temperature=0.7, json_mode=True)
        data = self._clean_json(response)
        self._save_json("01_blueprint.json", data)
        return data

    # --- STAGE 2: SCRIPTING ---
    def stage_2_script(self, blueprint: dict) -> list:
        self.logger.info("✍️ Stage 2: Writing Script...")
        system_prompt = PROMPTS["STAGE_2_SCRIPT"].format(page_count=PIPELINE_SETTINGS["PAGE_COUNT"])
        user_input = f"Write the script using this blueprint: {json.dumps(blueprint)}"
        
        response = self.llm.generate(system_prompt, user_input, temperature=0.7, json_mode=True)
        data = self._clean_json(response)
        
        if isinstance(data, dict):
             for key in ["pages", "script", "content"]:
                 if key in data and isinstance(data[key], list):
                     data = data[key]
                     break
        
        self._save_json("02_script.json", data)
        return data

    # --- STAGE 3: VISUAL PROMPTING ---
    def stage_3_visual_prompts(self, script: list, blueprint: dict) -> list:
        self.logger.info("🎨 Stage 3: Engineering Consistent Visuals...")
        
        # Extract Anchors
        style_anchor = blueprint.get("visual_style", "Cinematic digital art")
        char_anchor = f"{blueprint['protagonist']['name']}, {blueprint['protagonist']['consistent_char_prompt']}"
        
        # Inject anchors into System Prompt
        system_prompt = PROMPTS["STAGE_3_VISUALS"].format(
            style=style_anchor,
            char_desc=char_anchor,
            script_json=json.dumps(script) # Pass the full script with visual_scenes
        )
        
        user_input = "Generate the image prompts now."
        response = self.llm.generate(system_prompt, user_input, temperature=0.6, json_mode=True)
        data = self._clean_json(response)
        
        if isinstance(data, dict):
             for key in ["prompts", "images", "pages"]:
                 if key in data and isinstance(data[key], list):
                     data = data[key]
                     break

        # Merge Logic
        final_plan = []
        for i, page_item in enumerate(data):
            original_page = next((p for p in script if p.get('page') == page_item.get('page')), None)
            text_content = original_page['text'] if original_page else "..."
            
            final_plan.append({
                "page": page_item.get('page', i+1),
                "text": text_content,
                "image_prompt": page_item.get('image_prompt', "")
            })
            
        self._save_json("03_production_plan.json", final_plan)
        return final_plan

    # --- STAGE 4: RENDERING ---
    def stage_4_render(self, production_plan: list, blueprint: dict):
        self.logger.info("🏭 Stage 4: Rendering Assets (Parallel)...")
        
        # Extract Negative Prompt from Blueprint
        neg_prompt = blueprint.get("negative_prompt", "low quality, text, blurry")

        def render_page(page_data):
            pid = page_data['page']
            
            # 1. Audio
            audio_path = self.audio_dir / f"page_{pid:02d}.mp3"
            self.narrator.generate(page_data['text'], audio_path)
            
            # 2. Image (Passing Negative Prompt now)
            img_path = self.img_dir / f"page_{pid:02d}.webp"
            self.painter.generate(
                prompt=page_data['image_prompt'], 
                output_path=img_path, 
                negative_prompt=neg_prompt,
                aspect_ratio=PIPELINE_SETTINGS["IMAGE_ASPECT_RATIO"]
            )
            return pid

        with ThreadPoolExecutor(max_workers=PIPELINE_SETTINGS["MAX_WORKERS"]) as executor:
            list(executor.map(render_page, production_plan))
            
        self.logger.info("✅ Assets Generated.")

    # --- STAGE 5: ASSEMBLY ---
    def stage_5_assemble(self, production_plan: list, blueprint: dict):
        self.logger.info("📦 Stage 5: Final Assembly...")
        
        title = blueprint.get('title', 'Storybook')
        md_content = f"# {title}\n\n"
        plain_text_content = f"TITLE: {title}\n\n"
        
        for page in sorted(production_plan, key=lambda x: x['page']):
            pid = page['page']
            text = page['text']
            
            md_content += f"## Page {pid}\n"
            md_content += f"![Scene]({self.img_dir.name}/page_{pid:02d}.webp)\n"
            md_content += f"**Audio:** [Listen]({self.audio_dir.name}/page_{pid:02d}.mp3)\n\n"
            md_content += f"> {text}\n\n---\n\n"
            
            plain_text_content += f"Page {pid}: {text}\n\n"
            
        self._save_file("storybook.md", md_content)
        self._save_file("story.txt", plain_text_content)
        self.logger.info(f"✨ COMPLETED. Output in: {self.base_dir}")

    # --- MAIN RUN METHOD ---
    def run(self, user_prompt: str):
        # 1. Cleanup
        self._cleanup()
        
        # 2. Pipeline Execution
        constraints = self.stage_0_analyze(user_prompt)
        blueprint = self.stage_1_blueprint(constraints)
        script = self.stage_2_script(blueprint)
        production_plan = self.stage_3_visual_prompts(script, blueprint)
        
        # Pass blueprint to render stage for negative prompts
        self.stage_4_render(production_plan, blueprint) 
        
        self.stage_5_assemble(production_plan, blueprint)

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # --- USER INPUT ---
    USER_PROMPT = "a girl in Delhi, India was burnt alive by her in laws family for the demand of dowry by her in laws and husband"
    OUTPUT_FOLDER = "/root/Desktop/yt-automation/picturebook-1/"
    
    # --- EXECUTE ---
    generator = StorybookGenerator(output_dir=OUTPUT_FOLDER)
    generator.run(USER_PROMPT)
