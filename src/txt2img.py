import base64
import random
from datetime import datetime
from io import BytesIO
from typing import Callable, Optional

import requests
import torch
from PIL import Image

import utils


class Txt2Img:
    async def generate(
        self,
        prompt: str,
        generation_settings: dict[str, int | float],
        neg_prompt: Optional[str] = None,
        model: str = "AnythingV5.safetensors [a1535d0a42]",
        loras: list[utils.Lora] = [],
        dimensions: tuple[int, int] = (512, 768),
        callback: Optional[Callable[[int, torch.Tensor, dict], None]] = None,
        save: bool = False,
    ) -> tuple[int, list[Image.Image]]:
        prompt, neg_prompt = utils.prepare_prompt(prompt, loras, neg_prompt)

        generation_settings = generation_settings.copy()

        seed = random.randrange(1000000000)

        settings = {
            "sd_model_checkpoint": model,
            "sd_vae": "Automatic",
        }

        n_iter = generation_settings.pop("n_iter")
        batch_size = 1
        for i in range(2, 4):
            d, m = divmod(n_iter, i)
            if m == 0:
                batch_size = i
                n_iter = d
                break

        print()
        print(str(datetime.now()).split(".")[0])
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
            "override_settings": settings,
            "resize_mode": 2,
            "inpaint_full_res_padding": 32,
            "sampler_name": "DPM++ 2M Karras",
            "sampler_index": "DPM++ 2M Karras",
            "seed": seed,
        }

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
