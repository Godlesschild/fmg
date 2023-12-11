import gc
import os
import re
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple, Optional

import requests
import telegram
import tomli
from more_itertools import flatten
from PIL import Image
from telegram import InputMediaPhoto, Message
from telegram.ext import ContextTypes, ConversationHandler

ASSETS_DIR = Path(".", "stable-diffusion-webui")
MODELS_DIR = ASSETS_DIR / "models"

NEGATIVE_EMBEDDINGS_DIR = ASSETS_DIR / "embeddings"
DIFFUSION_DIR = MODELS_DIR / "Stable-diffusion"
LORA_DIR = MODELS_DIR / "lora"

PRE_PROMPT_PATH = Path(".", "pre_prompt.txt")
LORA_PATH = Path(".", "loras.txt")

URL = "http://127.0.0.1:7861"


class Lora(NamedTuple):
    name: str
    weight: float
    trigger_words: Iterable[str]

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


def get_config() -> dict[str, Any]:
    with open("config.toml", "rb") as file:
        toml = tomli.load(file)

    return toml


def styles() -> list[Lora]:
    requests.post(url=f"{URL}/sdapi/v1/refresh-loras")

    styles = set()
    lora_pattern = re.compile(r"\(<(\w+):([0-9.]+)>;\s?(.*?)\)")

    config = get_config()

    lora_lines = config["loras"]["style_loras"]

    for lora_str in lora_lines:
        match = lora_pattern.search(lora_str)

        if match is None:
            continue

        styles.add(Lora(match.group(1), float(match.group(2)), tuple(match.group(3).split(", "))))

    return list(styles)


def models() -> list[str]:
    requests.post(url=f"{URL}/sdapi/v1/refresh-checkpoints")

    response = requests.get(url=f"{URL}/sdapi/v1/sd-models").json()
    return [model["title"] for model in response]


def loras() -> list[Lora]:
    requests.post(url=f"{URL}/sdapi/v1/refresh-loras")

    loras = set()
    lora_pattern = re.compile(r"\(<(\w+):([0-9.]+)>;\s?(.*?)\)")

    config = get_config()

    lora_lines = config["loras"]["other_loras"]

    for lora_str in lora_lines:
        match = lora_pattern.search(lora_str)

        if match is None:
            continue

        loras.add(Lora(match.group(1), float(match.group(2)), tuple(match.group(3).split(", "))))

    return list(loras)


def prepare_prompt(prompt: str, loras: list[Lora], neg_prompt: Optional[str] = None) -> tuple[str, str]:
    lora_triggers = ""
    lora_keywords = ""
    if len(loras) > 0:
        lora_triggers = list(flatten([lora.trigger_words for lora in loras]))
        lora_triggers = ", " + ", ".join(lora_triggers)

        lora_keywords = [f"<lora:{lora.name}:{lora.weight}>" for lora in loras]
        lora_keywords = " " + " ".join(lora_keywords)

    pre = get_config()["txt2img_pre_prompt"]["pre_prompt"]
    if pre != "":
        pre = pre + ", "

    prompt = re.sub(r"(\{*masterpiece\}*)|(\{*best quality\}*)", "", prompt).strip("., ")
    prompt = pre + prompt + lora_triggers + lora_keywords

    neg_prompt = "" if neg_prompt is None else neg_prompt
    neg_prompt = " ".join(neg_embeddings()) + " " + neg_prompt

    return (prompt, neg_prompt)


async def send_prompts(message: Message, tokens: Iterator[str]):
    list_pattern = re.compile(r"\d+\.")
    prompt = ""

    # skip first list number
    for token in tokens:
        if "." in token:
            break

    for token in tokens:
        prompt += token

        match = list_pattern.search(prompt)
        if match is not None:
            await message.edit_text(prompt[: match.start()], reply_markup=None)

            prompt = ""
            message = await message.reply_text("Generating prompt...")
            continue

        if token.strip() != "":
            await message.edit_text(prompt, reply_markup=None)
