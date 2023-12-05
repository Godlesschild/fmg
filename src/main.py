import os
import shlex
import subprocess
import sys
from enum import Enum
from io import BytesIO
from time import sleep
from typing import Optional
from warnings import filterwarnings

import nest_asyncio
import requests
from more_itertools import chunked_even
from PIL import Image
from telegram import InlineKeyboardButton as IKB
from telegram import InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    PicklePersistence,
    filters,
)
from telegram.warnings import PTBUserWarning

import img2img_conv
import txt2img_conv
import txt2prompt_conv
import utils
from img2img_conv import IMG2IMG, IMG2IMG_STATE
from pix2pix import Pix2Pix
from txt2img_conv import TXT2IMG, TXT2IMG_STATE
from txt2prompt_conv import TXT2PROMPT_STATE

filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)
nest_asyncio.apply()


AUTOMATIC1111_DIR = utils.ASSETS_DIR.absolute()
AUTOMATIC1111 = subprocess.Popen(
    f"{AUTOMATIC1111_DIR}\\webui-user.bat",
    cwd=AUTOMATIC1111_DIR,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
)


server_started = False
while not server_started:
    sleep(7.5)
    try:
        requests.get(f"{utils.URL}/sdapi/v1/sd-models").json()
        server_started = True
    except:
        pass


PIX2PIX = Pix2Pix()


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[utils.STATE]:
    if context.user_data is None:
        return None

    context.user_data["settings"] = {}
    context.user_data["loras"] = []
    context.user_data["prompt"] = None
    context.user_data["neg_prompt"] = None
    context.user_data["genre"] = None
    context.user_data["seed"] = None

    context.user_data["model"] = next(model for model in utils.models() if "anylora" in model)
    context.user_data["style"] = None

    return await start(update, context)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[utils.STATE]:
    if update.message is None or context.user_data is None:
        return None

    buttons = [
        [IKB("txt2prompt", callback_data="txt2prompt")],
        [IKB("txt2img", callback_data="txt2img"), IKB("undress", callback_data="undress")],
        [IKB("img2img", callback_data="img2img"), IKB("pix2pix", callback_data="pix2pix")],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    await update.message.reply_text("Hi!!!!", reply_markup=keyboard)

    return utils.STATE.START


async def start_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[Enum]:
    query = update.callback_query

    if query is None or context.user_data is None or query.message is None:
        return None

    await query.answer()

    data = query.data
    match data:
        case "txt2prompt":
            await query.edit_message_text("Enter prompt amount.")
            return TXT2PROMPT_STATE.GET_AMOUNT
        case "txt2img":
            await query.edit_message_text("TXT2IMG", reply_markup=txt2img_conv.START_KEYBOARD)

            return TXT2IMG_STATE.START
        case "img2img":
            context.user_data["model"] = None

            models = list(IMG2IMG.PER_MODEL_SETTINGS.keys())
            keyboard = ReplyKeyboardMarkup(
                list(chunked_even(models, 3)), one_time_keyboard=True, resize_keyboard=True
            )

            default_keyboard = InlineKeyboardMarkup([[IKB("Default", callback_data="Default")]])

            await query.edit_message_text("Choose default model:", reply_markup=default_keyboard)
            await query.message.reply_text("Or choose yourself.", reply_markup=keyboard)

            return IMG2IMG_STATE.MODEL
        case "pix2pix":
            await query.edit_message_text("pix2pix")
            # TODO return IMG2IMG_STATE.GET_IMAGE
        case "undress":
            await query.edit_message_text("undress")
            # TODO return IMG2IMG_STATE.GET_IMAGE


async def undress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[utils.STATE]:
    if update.message is None or context.user_data is None:
        return

    file = await update.message.effective_attachment[-1].get_file()  # type: ignore
    file_buf = BytesIO(await file.download_as_bytearray())

    image = Image.open(file_buf).convert("RGB")

    if max(image.size) > 1024:
        factor = 1024 / max(image.size)
        new_size = utils.round_size((image.width * factor, image.height * factor))
        image = image.resize(new_size, resample=Image.Resampling.LANCZOS)

    images = await PIX2PIX.generate("make her naked with small breasts", image, gen_num=1)
    await PIX2PIX.free()

    await utils.send_images(images, context, update.message, caption="")

    unmasked_prompt = '[txt2mask method="clipseg" mode="add" sketch_alpha=0.0 show unload_model precision=100.0 padding=30.0 smoothing=20.0 stamp_method="stretch" stamp_x=0.0 stamp_y=0.0]face[/txt2mask]'
    prompt = "1girl, (((small breasts))), nude body, nude breasts, nipples"

    mask = await IMG2IMG.create_mask(images[0], unmasked_prompt)

    images = await IMG2IMG.generate(images, prompt, mask=mask)
    await utils.send_images(images, context, update.message, caption="")

    # await progress_message.delete()


if __name__ == "__main__":
    persistence = PicklePersistence("./persistence/persistence", update_interval=30, single_file=False)
    app = (
        Application.builder()
        .token(utils.TOKEN)
        .concurrent_updates(False)
        .persistence(persistence)
        .arbitrary_callback_data(True)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            utils.STATE.START: [CallbackQueryHandler(start_buttons)],
            # TXT2PROMPT
            TXT2PROMPT_STATE.GET_AMOUNT: [
                MessageHandler(filters.TEXT, txt2prompt_conv.amount_txt2prompt),
            ],
            TXT2PROMPT_STATE.PROMPT: [
                MessageHandler(filters.TEXT, txt2prompt_conv.txt2prompt),
            ],
            # TXT2IMG
            TXT2IMG_STATE.START: [
                CallbackQueryHandler(txt2img_conv.start_txt2img),
            ],
            TXT2IMG_STATE.MODEL: [
                CallbackQueryHandler(txt2img_conv.model_txt2img),
            ],
            TXT2IMG_STATE.STYLE: [
                CallbackQueryHandler(txt2img_conv.style_txt2img),
            ],
            TXT2IMG_STATE.LORAS: [
                CallbackQueryHandler(txt2img_conv.loras_txt2img),
            ],
            TXT2IMG_STATE.SETTINGS: [
                MessageHandler(filters.TEXT, txt2img_conv.settings_txt2img),
                CallbackQueryHandler(txt2img_conv.default_settings_txt2img),
            ],
            TXT2IMG_STATE.PROMPT: [
                MessageHandler(filters.TEXT, txt2img_conv.txt2img),
                CallbackQueryHandler(txt2img_conv.prompt_back),
            ],
            # IMG2IMG
            IMG2IMG_STATE.MODEL: [
                MessageHandler(filters.TEXT, img2img_conv.model_img2img),
                CallbackQueryHandler(img2img_conv.default_model_img2img),
            ],
            IMG2IMG_STATE.STYLE: [
                MessageHandler(filters.TEXT, img2img_conv.style_img2img),
            ],
            IMG2IMG_STATE.GET_PROMPT: [
                MessageHandler(filters.TEXT, img2img_conv.prompt_img2img),
                CallbackQueryHandler(img2img_conv.default_prompt_img2img),
            ],
            IMG2IMG_STATE.GET_IMAGE: [
                MessageHandler(filters.PHOTO, img2img_conv.get_image),
            ],
            IMG2IMG_STATE.PROMPT: [
                MessageHandler(filters.TEXT, img2img_conv.img2img),
            ],
        },  # type: ignore
        fallbacks=[
            CommandHandler("start", start),
            CommandHandler("restart", restart),
        ],
        allow_reentry=True,
        name="conv_handler",
        persistent=True,
    )

    app.add_handler(conv_handler)

    app.run_polling(allowed_updates=Update.ALL_TYPES)
