import base64
import gc
import os
import re
from datetime import datetime
from enum import Enum
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import requests
import telegram
from compel import Compel, DiffusersTextualInversionManager
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image
from telegram import InlineKeyboardMarkup, InputMediaPhoto, Message
from telegram.ext import ContextTypes, ConversationHandler
from torch import FloatTensor

ASSETS_DIR = Path(".", "stable-diffusion-webui")
MODELS_DIR = ASSETS_DIR / "models"

NEGATIVE_EMBEDDINGS_DIR = ASSETS_DIR / "embeddings"
DIFFUSION_DIR = MODELS_DIR / "Stable-diffusion"
LORA_DIR = MODELS_DIR / "lora"

GENRE_PROMPTS_PATH = Path(".", "genre_prompts.txt")

URL = "http://127.0.0.1:7861"


with open("credentials.txt", "r") as file:
    TOKEN = file.readline()
    MY_ID = file.readline()
    CHANNEL_ID = file.readline()
    ALLOWED_IDS = {MY_ID} | set(file.readline().split())


class Lora(NamedTuple):
    name: str
    weight: float
    trigger_words: tuple[str]

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False

        return self.name == other.name and self.weight == other.weight  # type: ignore

    def __str__(self) -> str:
        return self.name


class STATE(Enum):
    START = 1
    END = ConversationHandler.END


async def _embed_prompt(
    prompt: str, proc: Compel, append: list[str] = [], prepend: list[str] = []
) -> FloatTensor:
    prompt = prompt.strip("., ")
    prompt = f"{', '.join(prepend)}, {prompt}, {',' .join(append)}"
    prompt = prompt.strip("., ")

    return proc(prompt)


async def _load_embeddings(pipe: DiffusionPipeline) -> list[str]:
    neg_embeds = []

    for embedding in neg_embeddings():
        token = embedding.split(".")[0]
        try:
            pipe.load_textual_inversion(
                NEGATIVE_EMBEDDINGS_DIR,
                weight_name=embedding,
                token=token,
            )
        except ValueError:
            pass

        neg_embeds.append(token)

    return neg_embeds


def neg_embeddings() -> Iterator[str]:
    neg_embeddings = os.listdir(NEGATIVE_EMBEDDINGS_DIR)
    for embedding in neg_embeddings[::-1]:
        if embedding.endswith("disabled"):
            continue

        yield embedding


async def embed(
    pipe: DiffusionPipeline, prompt: str, neg_prompt: str, embeddings: Optional[int] = None
) -> tuple[FloatTensor, FloatTensor]:
    neg_embeds = (await _load_embeddings(pipe))[:embeddings]

    textual_inversion_manager = DiffusersTextualInversionManager(pipe)
    compel_proc = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        textual_inversion_manager=textual_inversion_manager,
    )

    prompt_embeds = await _embed_prompt(prompt, compel_proc)

    neg_prompt_embeds = await _embed_prompt(neg_prompt, compel_proc, neg_embeds)

    return (prompt_embeds, neg_prompt_embeds)


async def save_images(images: list[Image.Image], generator_type: type):
    date = str(datetime.now().date())
    folder_path = Path(".", "images", date)
    os.makedirs(folder_path, exist_ok=True)

    generator_name = generator_type.__name__

    # get current index
    filenames = [filename for filename in os.listdir(folder_path) if generator_name in filename]
    cur_num = 0 if len(filenames) == 0 else max(int(filename.split("-")[-1]) for filename in filenames)

    for i, image in enumerate(images, start=1):
        image.save(folder_path / f"{generator_name}-{cur_num+1}-{i}.png")


async def send_images(
    images: list[Image.Image],
    context: ContextTypes.DEFAULT_TYPE,
    message: Message,
    caption: str,
):
    result = []
    for i in range(len(images)):
        bio = BytesIO()
        images[i].save(bio, "PNG")
        bio.seek(0)
        result.append(InputMediaPhoto(bio))
        gc.collect()

    await context.bot.send_media_group(
        message.chat_id,
        result,
        caption=caption,
        parse_mode=telegram.constants.ParseMode.HTML,
    )


def round_up_64(num: int | float) -> int:
    return 64 * ceil(num / 64)


def round_size(size: tuple[int, int] | tuple[float, float]) -> tuple[int, int]:
    return (round_up_64(size[0]), round_up_64(size[1]))


def image_to_base64(image: Image.Image) -> str:
    output_bytes = BytesIO()

    image.save(output_bytes, format="PNG")
    img_base64 = base64.b64encode(output_bytes.getvalue()).decode("utf-8")

    return img_base64


def downscale_image(image: Image.Image):
    if max(image.size) > 1024:
        factor = 1024 / max(image.size)
        new_size = round_size((image.width * factor, image.height * factor))
        image = image.resize(new_size, resample=Image.Resampling.LANCZOS)

    return image


def styles() -> list[Lora]:
    with open(GENRE_PROMPTS_PATH, "r") as file:
        style_pattern = re.compile(r"\(<(\w+):([0-9.]+)>;\s?(.*?)\)")

        styles = [
            Lora(style.group(1), float(style.group(2)), tuple(style.group(3).split(", ")))  # type: ignore
            for style in style_pattern.finditer(file.readline())
        ]

    return styles


def models() -> list[str]:
    response = requests.get(url=f"{URL}/sdapi/v1/sd-models").json()
    return [model["title"] for model in response]


def loras() -> list[Lora]:
    loras = set()

    with open(GENRE_PROMPTS_PATH, "r") as file:
        lora_pattern = re.compile(r"\(<(\w+):([0-9.]+)>;\s?(.*?)\)")

        file.readline()

        for line in file:
            if lora_pattern.search(line) is None:
                continue

            matches = lora_pattern.finditer(line)

            loras.update(
                Lora(lora.group(1), float(lora.group(2)), tuple(lora.group(3).split(", ")))  # type: ignore
                for lora in matches
            )

    return list(loras)


def genre_prompts() -> dict[str, list[str]]:
    genre_prompts: dict[str, list[str]] = {}
    with open(GENRE_PROMPTS_PATH, "r") as file:
        genre_pattern = re.compile(r"^([a-zA-Z\\/, ]+):$")
        lora_line_pattern = re.compile(r"\w+\s+lora -")

        for _ in range(3):
            file.readline()

        cur_genre = ""
        for line in file:
            line = line.strip()
            if len(line) == 0 or lora_line_pattern.match(line):
                continue

            if genre_match := genre_pattern.match(line):
                cur_genre = genre_match.group(1)
                genre_prompts[cur_genre] = []
            else:
                genre_prompts[cur_genre].append(line)

    return genre_prompts
