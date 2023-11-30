import gc
import random
import re
from pathlib import Path
from typing import Optional

import llm

import utils

PRE_PROMPT_PATH = Path(".", "pre_prompt.txt")
GENRE_PROMPTS_PATH = Path(".", "genre_prompts.txt")


class Txt2Prompt:
    def __init__(self) -> None:
        self.model = None

    async def setup(self, model: Optional[str] = None):
        if model is None or model not in llm.get_model_aliases():
            model = "nous-hermes-llama2-13b"

        self.model = llm.get_model(model)

    async def free(self):
        del self.model
        gc.collect()

    async def generate(self, genre: str, amount: int) -> list[str]:
        if self.model is None:
            await self.setup()

        if self.model is None:
            raise RuntimeError("yaaay the type checker shut the fuck up")

        genre_prompts = utils.genre_prompts()
        if genre not in genre_prompts:
            raise RuntimeError(f"Unknown genre: {genre}")

        with open(PRE_PROMPT_PATH, "r") as file:
            pre_prompt = file.read()

        example_prompts = random.sample(genre_prompts[genre], min(5, len(genre_prompts[genre])))

        prompt = pre_prompt.replace("{EXAMPLE_PROMPTS}", "\n\n".join(example_prompts))
        prompt = prompt.replace("{AMOUNT}", str(amount))

        gc.collect()

        response = [
            re.sub(r"^\s*?[0-9a-zA-Z]\.|\)\s+", "", line.strip(".,- "))
            for line in self.model.prompt(prompt).text().splitlines()
            if len(line.strip()) > 0
        ]
        gc.collect()

        return response
