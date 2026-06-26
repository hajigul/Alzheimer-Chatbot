import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
GUI_PATH = PROJECT_ROOT / "src" / "gui.py"


def main():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(GUI_PATH)],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()