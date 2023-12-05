# fmg - A stable diffusion telegram bot

## Usage

1. Clone [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) into the repo folder.
2. Drop Loras and Models into their directories inside of the `stable-diffusion-webui` folder.
3. Models will be loaded automatically, but loras must be added to the `genre_prompts.txt` file.
4. Create a `credentials.txt` file and paste your bot token there.
5. Add your pre prompt to `pre_prompt.txt`. 
6. Add loras to `loras.txt`.
7. Run `py src/main.py` from the root repo directory.
