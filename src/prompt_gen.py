import asyncio

from gpt4all import GPT4All

import utils


class PromptGen:
    def __init__(self):
        self.model = None

    async def generate(self) -> str:
        if self.model is None:
            self.model = GPT4All("nous-hermes-llama2-13b.Q4_0.gguf")

        return await asyncio.to_thread(self.model.generate, self.get_prompt(), max_tokens=1000)

    @staticmethod
    def get_prompt() -> str:
        config = utils.get_config()["prompt_gen_prompt"]

        prompt = (
            "### Instruction:"
            "\n"
            f"{config['instruction'].strip()}"
            "\n"
            "\n"
            "### Input:"
            f"{config['context'].strip()}"
            "\n"
            "\n"
            "### Response:"
            "\n"
        )

        return prompt


PROMPT_GEN = PromptGen()
