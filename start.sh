#!/bin/bash

# Check if git installed and in PATH
if command -v git &> /dev/null; then
    echo "Git is installed."
else
    echo "Git is not installed, aborting.."
    exit 1
fi

echo

git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

echo

# Check if python and pip are installed
if command -v python3.10 &> /dev/null; then
    echo "Python is installed."
else
    echo "Python is not installed, aborting..."
    exit
fi


if ! python3.10 -m pip &> /dev/null; then
    echo "pip is not installed, run 'apt install python3-pip python3-venv'"
    echo "Aborting..."
    exit
fi

# Create venv
cd stable-diffusion-webui || exit
python3.10 -m venv venv
cd ..

# Activate venv
source "$(pwd)/stable-diffusion-webui/venv/bin/activate"
echo

# Set environment variables
if [ -z "$NO_GPU" ]; then
    export COMMANDLINE_ARGS="--xformers --api --nowebui"
else
    export COMMANDLINE_ARGS="--api --nowebui --no-half --skip-torch-cuda-test --use-cpu all"
    export TORCH_COMMAND="pip install torch torchvision"
fi

echo "torch command: $TORCH_COMMAND"
echo 

# Install required dependencies
python3.10 -m pip install -r requirements.txt
python3.10 setup.py

echo
echo All dependencies installed.
echo

# Start bot
echo Starting bot...
echo

python3.10 "src/main.py"

read -r -p "Press Enter to exit..."
