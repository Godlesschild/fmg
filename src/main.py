import platform
import shlex
import subprocess
from time import sleep
from typing import Optional
from warnings import filterwarnings

import nest_asyncio
import requests
from telegram import Update
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
from txt2img_conv import STATE

filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)
nest_asyncio.apply()


AUTOMATIC1111_DIR = utils.ASSETS_DIR.absolute()

AUTOMATIC1111_COMMAND = f"{AUTOMATIC1111_DIR}\\webui.bat"
if platform.system() != "Windows":
    AUTOMATIC1111_COMMAND = shlex.split(f"bash {AUTOMATIC1111_DIR}/webui.sh -f")

TOKEN = utils.get_config()["credentials"]["token"]


async def start(update: Update, context: Optional[ContextTypes.DEFAULT_TYPE]) -> Optional[STATE]:
    if update.message is None or context.user_data is None:
        return None

    context.user_data["settings"] = {}
    context.user_data["loras"] = []
    context.user_data["prompt"] = None
    context.user_data["neg_prompt"] = None
    context.user_data["genre"] = None
    context.user_data["seed"] = None

    context.user_data["model"] = next(model for model in utils.models())
    context.user_data["style"] = None

    await update.message.reply_text(txt2img_conv.START_TEXT, reply_markup=txt2img_conv.START_KEYBOARD)

    return STATE.START


if __name__ == "__main__":
    AUTOMATIC1111 = subprocess.Popen(
        AUTOMATIC1111_COMMAND,
        cwd=AUTOMATIC1111_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    server_started = False
    while not server_started:
        sleep(3)
        try:
            requests.get(f"{utils.URL}/sdapi/v1/sd-models").json()
            server_started = True
        except:
            pass

    persistence = PicklePersistence("./persistence/persistence", update_interval=30, single_file=False)
    app = (
        Application.builder()
        .token(TOKEN)
        .concurrent_updates(True)
        .persistence(persistence)
        .arbitrary_callback_data(True)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            STATE.START: [
                CallbackQueryHandler(txt2img_conv.start),
            ],
            STATE.MODEL: [
                CallbackQueryHandler(txt2img_conv.model),
            ],
            STATE.STYLE: [
                CallbackQueryHandler(txt2img_conv.style),
            ],
            STATE.LORAS: [
                CallbackQueryHandler(txt2img_conv.loras),
            ],
            STATE.SETTINGS: [
                MessageHandler(filters.TEXT, txt2img_conv.settings),
                CallbackQueryHandler(txt2img_conv.default_settings),
            ],
            STATE.PROMPT: [
                MessageHandler(filters.TEXT, txt2img_conv.generate),
                CallbackQueryHandler(txt2img_conv.prompt_back),
            ],
        },  # type: ignore
        fallbacks=[
            CommandHandler("start", start),
        ],
        allow_reentry=True,
        name="conv_handler",
        persistent=True,
    )

    app.add_handler(conv_handler)

    app.run_polling(allowed_updates=Update.ALL_TYPES)
