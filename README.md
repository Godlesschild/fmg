# fmg - A stable diffusion telegram bot

## Automatic Installation on Windows

1. Clone this repo (or download as a zip file).
2. Place your bot token into `config.toml`.
3. Run `start.bat` from Windows Explorer.
4. Drop SD models, loras and embeddings into their respective directories inside of the `stable-diffusion-webui` folder.
    * SD models and embeddings will be loaded automatically, but loras must also be added to `config.toml` with their weights and trigger words.

## Running

If you want to run on cpu only, set the NO_GPU environment variable.

### Windows

Run `start.bat` from Windows Explorer as normal.
