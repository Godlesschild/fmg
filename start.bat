@echo off

:: Check if git installed and in PATH
where git > nul 2>&1
if %errorlevel% equ 0 (
    echo Git is installed.
) else (
    echo Git is not installed, aborting...
    exit
)

echo.

git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

echo.

:: Create venv
cd stable-diffusion-webui
python -m venv venv
cd ..

:: Activate venv
set PYTHON="%CD%\stable-diffusion-webui\venv\Scripts\python.exe"
echo venv: %PYTHON%
echo.

:: Install required dependencies
%PYTHON% setup.py
%PYTHON% -m pip install -r requirements.txt

echo.
echo All dependencies installed.
echo.

:: Set environment variables
if defined NO_GPU (
    set COMMANDLINE_ARGS="--xformers --api --nowebui --no-half --skip-torch-cuda-test --use-cpu all"
) else (
    set COMMANDLINE_ARGS="--xformers --api --nowebui"
)

:: Start bot
echo Starting bot...
echo.

%PYTHON% src\main.py

pause
