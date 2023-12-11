# fmg - A stable diffusion telegram bot

## Usage

1. Clone [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) into the repo folder.
2. Drop lora and SD models into their respective directories inside of the `stable-diffusion-webui` folder.
3. Models and embeddings will be loaded automatically, but loras must be added to `config.toml` file.
4. Place your bot token into `config.toml`.
5. Run `py src/main.py` from the root repo directory.
