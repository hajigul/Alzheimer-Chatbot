import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Starting 02_train_model.py...", flush=True)

from src.training import main

print("Imported src.training successfully.", flush=True)

if __name__ == "__main__":
    main()