import base64
import random
import re
from io import BytesIO
from typing import Callable, Optional

import requests
import torch
from more_itertools import flatten
from PIL import Image

import utils


class Txt2Img:
    AUTO_PROMPTS = "((masterpiece)), ((best quality)), ((sexy)), (((nsfw))), ((perfect face)), (((perfect anatomy))), ((naked))"

    # AUTO_LORA = "<lora:BreastInClass:0.3> <lora:MoreDetails:0.5> <lora:innievag:0.5>"
    AUTO_LORA = " <lora:BreastInClass:0.3> <lora:innievag:0.5>"

    PER_STYLE_SETTINGS = {
        # "anime_tarot": {"width": 512, "height": 768},
        # "takes_off": {"width": 512, "height": 768},
        # "botw": {"width": 512, "height": 768},
        # "arcane": {"width": 512, "height": 768},
    }

    async def generate(
        self,
        prompt: str,
        generation_settings: dict[str, int | float],
        neg_prompt: Optional[str] = None,
        model: str = "AnythingV5.safetensors [a1535d0a42]",
        loras: list[utils.Lora | None] = [],
        dimensions: tuple[int, int] = (512, 768),
        callback: Optional[Callable[[int, torch.Tensor, dict], None]] = None,
        save: bool = False,
    ) -> tuple[int, list[Image.Image]]:
        if len(loras) > 0:
            lora_triggers = (
                ", ".join(flatten([lora.trigger_words for lora in loras if lora is not None])) + ", "
            )
            loras_kw = " ".join([f"<lora:{lora.name}:{lora.weight}>" for lora in loras if lora is not None])
        else:
            lora_triggers = ""
            loras_kw = ""

        prompt = re.sub(r"(\{*masterpiece\}*)|(\{*best quality\}*)", "", prompt).strip("., ")
        prompt = f"{self.AUTO_PROMPTS}{lora_triggers}{prompt} {loras_kw}{self.AUTO_LORA}"

        neg_prompt = "" if neg_prompt is None else neg_prompt
        neg_prompt = " ".join(utils.neg_embeddings()) + " " + neg_prompt

        seed = random.randrange(1000000000)

        settings = {
            "sd_model_checkpoint": model,
        }

        n_iter = generation_settings.pop("n_iter", 3)
        batch_size = 1
        for i in range(2, 4):
            d, m = divmod(n_iter, i)
            if m == 0:
                batch_size = i
                n_iter = d
                break

        print()
        print()
        print(prompt)
        print()
        print(batch_size, n_iter)
        print()

        request = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "width": dimensions[0],
            "height": dimensions[1],
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": 30,
            "cfg_scale": 7,
            "override_settings": settings,
            "resize_mode": 2,
            "inpaint_full_res_padding": 32,
            "sampler_name": "DPM++ 2M Karras",
            "sampler_index": "DPM++ 2M Karras",
            "seed": seed,
        }

        [
            request.update(self.PER_STYLE_SETTINGS.get(style.name, {}))
            for style in loras
            if style in utils.styles()
        ]

        request.update(generation_settings)

        response = requests.post(f"{utils.URL}/sdapi/v1/txt2img", json=request)

        if response.status_code == 500:
            raise RuntimeError("Request error")

        result = []
        for result_image in response.json()["images"]:
            result_image = result_image.split(",", 1)[-1]
            result.append(Image.open(BytesIO(base64.b64decode(result_image))))

        # save images
        if save:
            await utils.save_images(result, type(self))

        return (seed, result)
