from io import BytesIO
from PIL import Image
from enum import Enum
from typing import Optional

from more_itertools import chunked_even
from telegram import InlineKeyboardButton as IKB
from telegram import InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import ContextTypes
from img2img import Img2Img
from txt2img import Txt2Img

import utils


TXT2IMG = Txt2Img()
IMG2IMG = Img2Img()


class IMG2IMG_STATE(Enum):
    MODEL = 31
    STYLE = 32
    GET_PROMPT = 33
    GET_IMAGE = 34
    PROMPT = 35
    END = utils.STATE.START


async def model_img2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[IMG2IMG_STATE]:
    if update.message is None or update.message.text is None or context.user_data is None:
        return

    models = list(IMG2IMG.PER_MODEL_SETTINGS.keys())
    if update.message.text not in models:
        await update.message.reply_text("Please choose from the keyboard.")
        return None

    context.user_data["model"] = update.message.text

    styles = [style.name for style in utils.styles()]
    keyboard = ReplyKeyboardMarkup([["None"], *chunked_even(styles, 3)])

    await update.message.reply_text(
        "Choose style.",
        reply_markup=keyboard,
    )
    return IMG2IMG_STATE.STYLE


async def default_model_img2img(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> Optional[IMG2IMG_STATE]:
    if update.callback_query is None or update.callback_query.message is None:
        return

    await update.callback_query.answer()

    styles = [style.name for style in utils.styles()]
    keyboard = ReplyKeyboardMarkup([["None"], *chunked_even(styles, 3)])

    await update.callback_query.message.reply_text(
        "Choose style.",
        reply_markup=keyboard,
    )

    return IMG2IMG_STATE.STYLE


async def style_img2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[IMG2IMG_STATE]:
    if update.message is None or update.message.text is None or context.user_data is None:
        return

    styles = [style.name for style in utils.styles()]
    if update.message.text not in styles:
        await update.message.reply_text("Please choose from the keyboard.")
        return None

    context.user_data["settings"]["style"] = next(
        style for style in utils.styles() if style.name == update.message.text
    )

    keyboard = InlineKeyboardMarkup([[IKB("Default", callback_data="Default")]])
    await update.message.reply_text(
        "Enter prompt or choose `Default`.",
        reply_markup=keyboard,
    )

    return IMG2IMG_STATE.GET_PROMPT


async def prompt_img2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[IMG2IMG_STATE]:
    if update.message is None or update.message.text is None or context.user_data is None:
        return

    context.user_data["settings"]["prompt"] = update.message.text

    await update.message.reply_text("Send image.")

    return IMG2IMG_STATE.GET_IMAGE


async def default_prompt_img2img(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> Optional[IMG2IMG_STATE]:
    if update.callback_query is None or update.callback_query.message is None or context.user_data is None:
        return

    await update.callback_query.answer()

    context.user_data["settings"]["prompt"] = "1girl, (((small breasts))), nude body, nude breasts, nipples"

    await update.callback_query.message.reply_text("Send image.")

    return IMG2IMG_STATE.GET_IMAGE


async def get_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[IMG2IMG_STATE]:
    if update.message is None or context.user_data is None:
        return

    file = await update.message.effective_attachment[-1].get_file()  # type: ignore
    file_buf = BytesIO(await file.download_as_bytearray())

    image = Image.open(file_buf).convert("RGB")
    image = utils.downscale_image(image)

    context.user_data["image"] = image

    keyboard = ReplyKeyboardMarkup([["Continue"]], one_time_keyboard=True, resize_keyboard=True)
    context.user_data["prompt_msg"] = await update.message.reply_text(
        context.user_data["settings"]["prompt"], reply_markup=keyboard
    )

    return IMG2IMG_STATE.PROMPT


async def img2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[IMG2IMG_STATE]:
    if update.message is None or context.user_data is None:
        return

    if update.message.text != "Continue":
        context.user_data["settings"]["prompt"] = update.message.text
        await context.user_data["prompt_msg"].delete()
        await update.message.reply_text(context.user_data["settings"]["prompt"])

    images = await IMG2IMG.generate([context.user_data["image"]], context.user_data["settings"]["prompt"])
    # TODO context.user_data["settings"]["style"]

    await utils.send_images(images, context, update.message)

    del context.user_data["prompt_msg"]

    return IMG2IMG_STATE.END
