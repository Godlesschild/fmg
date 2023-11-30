import gc
from typing import Callable, Optional

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_latent_upscale import (
    StableDiffusionLatentUpscalePipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from PIL import Image

import utils


class Pix2Pix:
    def __init__(self):
        self.pipe: Optional[DiffusionPipeline] = None
        self.upscaler: Optional[DiffusionPipeline] = None

    async def free(self):
        self.pipe = None
        self.upscaler = None
        gc.collect()

    async def setup(self, euler_a: bool = True):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
        ).to("cuda")
        gc.collect()

        self.pipe.safety_checker = None

        if not euler_a:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, use_karras_sigmas=True
            )

        self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler",
            torch_dtype=torch.float16,
        ).to("cuda")

    async def generate(
        self,
        prompt: str,
        image: Image.Image,
        gen_num: int = 3,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        num_inference_steps: int = 60,
        neg_prompt: Optional[str] = None,
        callback: Optional[Callable[[int, torch.Tensor, dict], None]] = None,
        save: bool = False,
    ) -> list[Image.Image]:
        if self.pipe is None or self.upscaler is None:
            await self.setup(euler_a=True)

        if self.pipe is None:
            raise RuntimeError("Just calming the typechecker")

        downscaled_image = image.resize(
            utils.round_size((image.width / 2, image.height / 2)), resample=Image.Resampling.LANCZOS
        )

        # embed prompts
        neg_prompt = neg_prompt if neg_prompt is not None else ""
        prompt_embeds, neg_prompt_embeds = await utils.embed(self.pipe, prompt, neg_prompt, 2)

        # generate images
        gc.collect()
        low_res = self.pipe(  # type: ignore
            image=downscaled_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=gen_num,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            callback=callback,
            callback_steps=5,
            output_type="latent",
        ).images
        gc.collect()

        settings = {"num_inference_steps": 40, "guidance_scale": 0, "image": low_res}

        images = self.upscaler(prompt="", **settings).images  # type: ignore

        # save images
        if save:
            await utils.save_images(images, type(self))

        return images
