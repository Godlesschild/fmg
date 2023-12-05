import random
import re
from enum import Enum
from typing import Optional

import telegram
from more_itertools import chunked_even
from telegram import InlineKeyboardButton as IKB
from telegram import InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import ContextTypes

import utils
from txt2img import Txt2Img
from utils import STATE

TXT2IMG = Txt2Img()


class TXT2IMG_STATE(Enum):
    START = 21
    MODEL = 22
    STYLE = 23
    LORAS = 24
    SETTINGS = 25
    PROMPT = 26
    END = STATE.START


START_KEYBOARD = InlineKeyboardMarkup(
    [
        *chunked_even(
            [
                IKB("Change model", callback_data="MODEL"),
                IKB("Change style", callback_data="STYLE"),
                IKB("Change loras", callback_data="LORAS"),
                IKB("Change settings", callback_data="SETTINGS"),
            ],
            2,
        ),
        [IKB("Generate", callback_data="PROMPT")],
    ]
)


async def start_txt2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2IMG_STATE]:
    query = update.callback_query

    if query is None or query.message is None or query.data is None or context.user_data is None:
        return

    await query.answer()

    message = ""
    keyboard = InlineKeyboardMarkup([[]])

    state = query.data
    match state:
        case "MODEL":
            message = "Choose model"

            selected = context.user_data["model"].split(".")[0]
            models = [model.split(".")[0] for model in utils.models()]

            for i, model in enumerate(models):
                if model != selected:
                    continue
                models[i] = "✅" + model

            keyboard = InlineKeyboardMarkup(
                list(chunked_even([IKB(model, callback_data=model) for model in models], 3))
                + [[IKB("⬅️", callback_data="BACK")]]
            )

            await query.edit_message_text(message)
            await query.edit_message_reply_markup(keyboard)

            return TXT2IMG_STATE.MODEL

        case "STYLE":
            message = "Choose style"

            selected = str(context.user_data["style"])
            styles = ["None"] + [str(style) for style in utils.styles()]

            for i, style in enumerate(styles):
                if style != selected:
                    continue
                styles[i] = "✅" + style

            styles = [IKB(style, callback_data=style.strip("✅")) for style in styles]
            keyboard = InlineKeyboardMarkup(
                [
                    *chunked_even(styles, 3),
                    [IKB("⬅️", callback_data="BACK")],
                ]
            )

            await query.edit_message_text(message)
            await query.edit_message_reply_markup(keyboard)

            return TXT2IMG_STATE.STYLE

        case "LORAS":
            message = "Choose loras"

            selected = context.user_data["loras"]

            buttons = [
                IKB("✅" + lora.name if lora in selected else lora.name, callback_data=lora.name)
                for lora in utils.loras()
            ]

            keyboard = InlineKeyboardMarkup(
                [
                    [IKB("Clear", callback_data="CLEAR")],
                    *chunked_even(buttons, 3),
                    [IKB("⬅️", callback_data="BACK")],
                ]
            )

            await query.edit_message_text(message)
            await query.edit_message_reply_markup(keyboard)

            return TXT2IMG_STATE.LORAS

        case "SETTINGS":
            message = "Enter generation settings:\n\n`n_iter`: 3,\n`guidance_scale`: 7\n`denoising_strength`: 1,\nnum_inference_steps: 30,\nOptional[`neg_prompt`]: ___"
            keyboard = InlineKeyboardMarkup(
                [[IKB("Default", callback_data="Default")], [IKB("⬅️", callback_data="BACK")]]
            )

            await query.edit_message_text(message)
            await query.edit_message_reply_markup(keyboard)

            return TXT2IMG_STATE.SETTINGS
        case "PROMPT":
            message = "Enter prompt."
            keyboard = InlineKeyboardMarkup([[IKB("⬅️", callback_data="BACK")]])

            await query.edit_message_text(message)
            await query.edit_message_reply_markup(keyboard)

            return TXT2IMG_STATE.PROMPT


async def model_txt2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2IMG_STATE]:
    query = update.callback_query
    if query is None or query.message is None or query.data is None or context.user_data is None:
        return

    await query.answer()

    if query.data == "BACK":
        await query.edit_message_text("TXT2IMG")
        await query.edit_message_reply_markup(START_KEYBOARD)

        return TXT2IMG_STATE.START

    context.user_data["model"] = next(model for model in utils.models() if query.data.strip("✅") in model)

    selected = context.user_data["model"].split(".")[0]
    models = [model.split(".")[0] for model in utils.models()]

    for i, model in enumerate(models):
        if model != selected:
            continue
        models[i] = "✅" + model

    keyboard = InlineKeyboardMarkup(
        list(chunked_even([IKB(model, callback_data=model) for model in models], 3))
        + [[IKB("⬅️", callback_data="BACK")]]
    )

    await query.edit_message_reply_markup(keyboard)


async def style_txt2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2IMG_STATE]:
    query = update.callback_query
    if query is None or query.message is None or query.data is None or context.user_data is None:
        return

    await query.answer()

    if query.data == "BACK":
        await query.edit_message_text("TXT2IMG")
        await query.edit_message_reply_markup(START_KEYBOARD)

        return TXT2IMG_STATE.START

    context.user_data["style"] = next(
        (style for style in utils.styles() if str(style) == query.data.strip("✅")), None
    )

    selected = str(context.user_data["style"])
    styles = ["None"] + [str(style) for style in utils.styles()]

    for i, style in enumerate(styles):
        if style != selected:
            continue
        styles[i] = "✅" + style

    styles = [IKB(style, callback_data=style.strip("✅")) for style in styles]
    none = styles[0]
    del styles[0]

    keyboard = InlineKeyboardMarkup(
        [
            [none],
            *chunked_even(styles, 3),
            [IKB("⬅️", callback_data="BACK")],
        ]
    )

    await query.edit_message_reply_markup(keyboard)


async def loras_txt2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2IMG_STATE]:
    query = update.callback_query
    if query is None or query.message is None or query.data is None or context.user_data is None:
        return

    await query.answer()

    if query.data == "BACK":
        await query.edit_message_text("TXT2IMG")
        await query.edit_message_reply_markup(START_KEYBOARD)

        return TXT2IMG_STATE.START

    if query.data == "CLEAR":
        context.user_data["loras"] = []
    else:
        name = query.data
        pressed = next(lora for lora in utils.loras() if lora.name == name.strip("✅"))

        if "✅" in name:
            context.user_data["loras"].remove(pressed)
        else:
            context.user_data["loras"].append(pressed)

    selected = context.user_data["loras"]
    buttons = [
        IKB(
            "✅" + lora.name if lora in selected else lora.name,
            callback_data="✅" + lora.name if lora in selected else lora.name,
        )
        for lora in utils.loras()
    ]

    keyboard = InlineKeyboardMarkup(
        [
            [IKB("Clear", callback_data="CLEAR")],
            *chunked_even(buttons, 3),
            [IKB("⬅️", callback_data="BACK")],
        ]
    )

    await query.edit_message_reply_markup(keyboard)

    return TXT2IMG_STATE.LORAS


async def settings_txt2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2IMG_STATE]:
    if update.message is None or update.message.text is None or context.user_data is None:
        return

    pattern = re.compile(r"^([0-9]+)\n([0-9.]+)\n([0-9.]+)\n([0-9]+)(\n(\w+))?")
    match = pattern.match(update.message.text)
    if match is None:
        await update.message.reply_text("Please reply in the correct format or choose `Default`.")
        return

    context.user_data["settings"]["n_iter"] = int(match.group(1))
    context.user_data["settings"]["guidance_scale"] = float(match.group(2))
    context.user_data["settings"]["denoising_strength"] = float(match.group(3))
    context.user_data["settings"]["num_inference_steps"] = int(match.group(4))
    context.user_data["neg_prompt"] = match.group(6)

    await update.message.reply_text("TXT2IMG", reply_markup=START_KEYBOARD)

    return TXT2IMG_STATE.START


async def default_settings_txt2img(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> Optional[TXT2IMG_STATE]:
    query = update.callback_query
    if query is None or query.message is None or query.data is None or context.user_data is None:
        return

    await query.answer()

    if query.data != "BACK":
        context.user_data["neg_prompt"] = None

    await query.edit_message_text("TXT2IMG")
    await query.edit_message_reply_markup(START_KEYBOARD)

    return TXT2IMG_STATE.START


async def prompt_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2IMG_STATE]:
    query = update.callback_query
    if query is None or query.message is None or query.data is None or context.user_data is None:
        return

    await query.edit_message_text("TXT2IMG")
    await query.edit_message_reply_markup(START_KEYBOARD)

    return TXT2IMG_STATE.START


async def txt2img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[TXT2IMG_STATE]:
    if update.message is None or update.message.text is None or context.user_data is None:
        return

    context.user_data["prompt"] = update.message.text

    style = context.user_data["style"]
    model = context.user_data["model"]
    prompt = context.user_data["prompt"]

    kwargs = {
        "prompt": prompt,
        "neg_prompt": context.user_data["neg_prompt"],
        "generation_settings": context.user_data["settings"],
    }
    if model is not None:
        kwargs["model"] = model

    kwargs["loras"] = context.user_data["loras"]

    if style is not None:
        kwargs["loras"].append(style)

    seed, images = await TXT2IMG.generate(**kwargs)

    style = style.name if style is not None else "none"
    model = model.split(".")[0] if model is not None else "none"
    caption = (
        f'✿ seed: <a href="https://t.me/+BMD17kF8MdJkMmEy">{seed}</a>\n'  # TODO remove link
        f"\n"
        f"⁺₊✧‧₊ style:  #{style}\n"
        f"‧₊❀˖° model: #{model}\n"
        f"\n"
        f"🎀 prompt:\n<pre>{prompt}</pre>"
    )

    await utils.send_images(images, context, update.message, caption)

    await update.message.reply_text("TXT2IMG", reply_markup=START_KEYBOARD)

    return TXT2IMG_STATE.START
