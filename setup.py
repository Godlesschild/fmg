import os
import sys
from pathlib import Path

if __name__ == "__main__":
    AUTOMATIC1111_DIR = Path(".", "stable-diffusion-webui")

    sys.path.insert(0, str(AUTOMATIC1111_DIR))
    from modules.launch_utils import prepare_environment

    prepare_environment()

    persistence_dir = Path(".", "persistence")
    os.makedirs(persistence_dir, exist_ok=True)
