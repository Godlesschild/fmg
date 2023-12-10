import base64
from io import BytesIO
from typing import Callable, Optional

import requests
import torch
from PIL import Image

import utils


class Img2Img:
    PER_MODEL_SETTINGS = {
        "AnythingV5.safetensors [a1535d0a42]": {
            "steps": 30,
            "cfg_scale": 10,
            "denoising_strength": 0.3,
        },
        "realisticVisionV51.safetensors [15012c538f]": {
            "steps": 150,
            "cfg_scale": 10,
            "denoising_strength": 0.4,
        },
    }

    AUTO_PROMPTS = "{{{{nude}}}}, {{{{nipples}}}}, {{{{{{naked}}}}}}, {{{{undressed}}}}, {{{{breasts}}}}"
    AUTO_LORA = "<MoreDetails:0.75>"

    async def create_mask(self, image: Image.Image, prompt: str) -> Image.Image:
        encoded_image = utils.image_to_base64(image)

        request = {
            "init_images": [encoded_image],
            "prompt": prompt,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 1,
        }

        response = requests.post(f"{utils.URL}/sdapi/v1/img2img", json=request)

        if response.status_code == 500:
            raise RuntimeError("Request error")

        response_images = response.json()["images"]
        mask = response_images[1].split(",", 1)[-1]
        mask = Image.open(BytesIO(base64.b64decode(mask))).convert("L")

        return mask

    async def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        mask: Optional[Image.Image] = None,
        model: str = "realisticVisionV51.safetensors [15012c538f]",
        gen_num: int = 3,
        guidance_scale: float = 10,
        denoising_strength: float = 0.45,
        steps: int = 60,
        neg_prompt: Optional[str] = None,
        callback: Optional[Callable[[int, torch.Tensor, dict], None]] = None,
        save: bool = False,
    ) -> list[Image.Image]:
        prompt = f"{self.AUTO_PROMPTS}, " + prompt.strip("., ") + f" {self.AUTO_LORA}"

        neg_prompt = neg_prompt if neg_prompt is not None else ""

        neg_prompt = " ".join(utils.neg_embeddings()) + " " + neg_prompt

        settings = {
            "sd_model_checkpoint": model,
        }

        request = {
            "init_images": [utils.image_to_base64(image) for image in images],
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            # "styles": [],
            "batch_size": len(images),
            "n_iter": gen_num,
            "steps": steps,
            "cfg_scale": guidance_scale,
            "denoising_strength": denoising_strength,
            "override_settings": settings,
            # "resize_mode": 3,
            # "refiner_checkpoint": ,
            # "refiner_switch_at": 0,
            "inpainting_fill": 1,
            "inpainting_mask_invert": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
            "sampler_name": "DPM++ 2M Karras",
            "sampler_index": "DPM++ 2M Karras",
            **(self.PER_MODEL_SETTINGS[model]),
        }

        if mask is not None:
            request["mask"] = utils.image_to_base64(mask)

        response = requests.post(f"{utils.URL}/sdapi/v1/img2img", json=request)

        if response.status_code == 500:
            raise RuntimeError("Request error")

        result = []
        for result_image in response.json()["images"]:
            result_image = result_image.split(",", 1)[-1]
            result.append(Image.open(BytesIO(base64.b64decode(result_image))))

        # save images
        if save:
            await utils.save_images(result, type(self))

        return result
