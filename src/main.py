import subprocess
from enum import Enum
from time import sleep
from typing import Optional
from warnings import filterwarnings

import nest_asyncio
import requests
from telegram import InlineKeyboardButton as IKB
from telegram import InlineKeyboardMarkup, Update
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

import txt2img_conv
import utils
from prompt_gen import PROMPT_GEN
from txt2img_conv import TXT2IMG_STATE

filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)
nest_asyncio.apply()


AUTOMATIC1111_DIR = utils.ASSETS_DIR.absolute()
AUTOMATIC1111 = subprocess.Popen(
    f"{AUTOMATIC1111_DIR}\\webui-user.bat",
    cwd=AUTOMATIC1111_DIR,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
)

TOKEN = utils.get_config()["credentials"]["token"]


server_started = False
while not server_started:
    sleep(3)
    try:
        requests.get(f"{utils.URL}/sdapi/v1/sd-models").json()
        server_started = True
    except:
        pass


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


async def start(update: Update, context: Optional[ContextTypes.DEFAULT_TYPE]) -> Optional[utils.STATE]:
    if update.message is None:
        return None

    buttons = [
        [IKB("Generate prompts", callback_data="prompt_gen")],
        [IKB("Generate image", callback_data="txt2img")],
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
        case "prompt_gen":
            await query.edit_message_text("Generating prompts...", reply_markup=None)

            tokens = PROMPT_GEN.generate()

            await utils.send_prompts(query.message, tokens)

            return await start(query, None)  # type: ignore

        case "txt2img":
            await query.edit_message_text("TXT2IMG", reply_markup=txt2img_conv.START_KEYBOARD)

            return TXT2IMG_STATE.START


if __name__ == "__main__":
    persistence = PicklePersistence("./persistence/persistence", update_interval=30, single_file=False)
    app = (
        Application.builder()
        .token(TOKEN)
        .concurrent_updates(False)
        .persistence(persistence)
        .arbitrary_callback_data(True)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            utils.STATE.START: [CallbackQueryHandler(start_buttons)],
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
