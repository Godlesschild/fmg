import os
import sys
from pathlib import Path

if __name__ == "__main__":
    AUTOMATIC1111_DIR = Path(".", "stable-diffusion-webui")

    sys.path.insert(0, str(AUTOMATIC1111_DIR))
    from modules.launch_utils import prepare_environment

    prepare_environment()

    webui_user_path = AUTOMATIC1111_DIR / "webui-user.bat"

    webui_user = """@echo off

    set PYTHON=
    set GIT=
    set VENV_DIR=
    set COMMANDLINE_ARGS=--xformers --api --nowebui

    call webui.bat"""

    with open(webui_user_path, "w") as file:
        file.writelines(webui_user)

    persistence_dir = Path(".", "persistence")
    os.makedirs(persistence_dir, exist_ok=True)
