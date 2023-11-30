from enum import Enum
from typing import Optional

from more_itertools import chunked_even
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import ContextTypes

import utils
from txt2prompt import Txt2Prompt

TXT2PROMPT = Txt2Prompt()


class TXT2PROMPT_STATE(Enum):
    GET_AMOUNT = 11
    PROMPT = 12
    END = utils.STATE.START


async def amount_txt2prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2PROMPT_STATE]:
    if update.message is None or update.message.text is None or context.user_data is None:
        return

    try:
        amount = int(update.message.text)
        assert 0 < amount <= 5
        context.user_data["amount"] = amount
    except:
        await update.message.reply_text("Enter 0 < amount <= 5.")
        return

    keyboard = ReplyKeyboardMarkup(
        list(chunked_even(utils.genre_prompts().keys(), 3)), one_time_keyboard=True, resize_keyboard=True
    )

    await update.message.reply_text("Choose genre.", reply_markup=keyboard)

    return TXT2PROMPT_STATE.PROMPT


async def txt2prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2PROMPT_STATE]:
    if update.message is None or update.message.text is None or context.user_data is None:
        return

    if update.message.text not in utils.genre_prompts().keys():
        await update.message.reply_text("Please choose from the keyboard.")
        return

    context.user_data["genre"] = update.message.text

    prompts = await TXT2PROMPT.generate(context.user_data["genre"], context.user_data["amount"])

    for prompt in prompts:
        await update.message.reply_text(prompt)

    return TXT2PROMPT_STATE.END
