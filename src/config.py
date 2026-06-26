from pathlib import Path


# Main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_FULL_CHAT_FILE = DATA_DIR / "full_Chat_data.csv"
RAW_TRAIN_FILE = DATA_DIR / "train.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.jsonl"
PROCESSED_VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.jsonl"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"

# Model folders
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_DIR = MODELS_DIR / "alzheimer-gpt2"

# Use smaller GPT-2 for testing.
# You can change this to "gpt2" later.
BASE_MODEL_NAME = "gpt2-medium"

# Training settings
DEFAULT_MAX_LENGTH = 256
DEFAULT_VALIDATION_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42

# Inference settings
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.3

# App settings
API_HOST = "127.0.0.1"
API_PORT = 8000