import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cardd_yolo11s_1024_final.pt"
MODEL_IMAGE_SIZE = 1024
UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables. Groq features will fail.")
